#!/usr/bin/env node
/**
 * 三連単オッズ(JSON)から「各艇の市場単勝確率＆合成単勝オッズ」を計算するスクリプト
 *
 * usage:
 *   node scripts/build_synth_single_from_3t.js [ODDS_ROOT] [OUT_CSV]
 *
 * example:
 *   node scripts/build_synth_single_from_3t.js public/odds/v1 data/derived/synth_single_from_3t.csv
 *
 * 前提:
 *   - ODDS_ROOT 配下に YYYY/MMDD/PID/<race>R.json という構成で3連単オッズが保存されている想定。
 *   - JSON内に「3連単の組み合わせとオッズ」が入っていること。
 *
 * !!! IMPORTANT !!!
 *   getTrifectaEntries() 内の「JSONから3連単オッズを取り出す部分」は、
 *   実際のJSON構造に合わせて必ず調整してください。
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const ODDS_ROOT = process.argv[2] || "public/odds/v1";
const OUT_CSV = process.argv[3] || "data/derived/synth_single_from_3t.csv";

// ---------- utils ----------
const z2 = (n) => String(n).padStart(2, "0");

async function* walkFiles(dir) {
  const dirents = await fsp.readdir(dir, { withFileTypes: true });
  for (const d of dirents) {
    const full = path.join(dir, d.name);
    if (d.isDirectory()) {
      yield* walkFiles(full);
    } else if (d.isFile()) {
      yield full;
    }
  }
}

/**
 * JSONから「三連単の(1着,2着,3着,オッズ)一覧」を取り出す
 *
 * ここはあなたのJSON構造に合わせて書き換えてOK。
 * 戻り値の形式:
 *   [{ h1: 1, h2: 2, h3: 3, odds: 12.3 }, ...]
 */
function getTrifectaEntries(json) {
  const entries = [];

  // ---- 例1: data.trifecta = [{ "comb": "1-2-3", "odds": "12.3" }, ...] の場合 ----
  if (Array.isArray(json.trifecta)) {
    for (const row of json.trifecta) {
      const comb = String(row.comb || "").trim();
      const oddsStr = String(row.odds ?? "").trim();
      const odds = Number(oddsStr);
      if (!comb || !Number.isFinite(odds) || odds <= 0) continue;

      const parts = comb.split("-").map((s) => Number(s));
      if (parts.length !== 3) continue;
      const [h1, h2, h3] = parts;
      if (![h1, h2, h3].every((v) => Number.isInteger(v) && v >= 1 && v <= 6)) continue;

      entries.push({ h1, h2, h3, odds });
    }
    return entries;
  }

  // ---- 例2: data.odds3t = { "1-2-3": "12.3", ... } の場合 ----
  if (json.odds3t && typeof json.odds3t === "object") {
    for (const [comb, oddsStrRaw] of Object.entries(json.odds3t)) {
      const combStr = String(comb || "").trim();
      const oddsStr = String(oddsStrRaw ?? "").trim();
      const odds = Number(oddsStr);
      if (!combStr || !Number.isFinite(odds) || odds <= 0) continue;

      const parts = combStr.split("-").map((s) => Number(s));
      if (parts.length !== 3) continue;
      const [h1, h2, h3] = parts;
      if (![h1, h2, h3].every((v) => Number.isInteger(v) && v >= 1 && v <= 6)) continue;

      entries.push({ h1, h2, h3, odds });
    }
    return entries;
  }

  // ---- TODO: 他の形式ならここに追記 ----
  // console.warn("Unknown 3T structure:", Object.keys(json));
  return entries;
}

/**
 * 1レース分の三連単オッズから、
 * 各艇の「市場単勝確率 p_mkt」と「合成単勝オッズ synth_odds」を計算
 */
function calcHeadProbsFrom3T(entries) {
  // head_sums[i] = 「i号艇頭の全通り 1/odds の合計」
  const head_sums = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };

  for (const { h1, h2, h3, odds } of entries) {
    if (!Number.isFinite(odds) || odds <= 0) continue;
    const w = 1 / odds;
    if (head_sums[h1] === undefined) continue;
    head_sums[h1] += w;
  }

  const totalSum = Object.values(head_sums).reduce((a, b) => a + b, 0);
  const result = [];

  if (totalSum <= 0) {
    // オッズが極端に欠けてるなど
    for (let head = 1; head <= 6; head++) {
      result.push({
        head,
        p_mkt: 0,
        synth_odds: null,
        raw_sum_inv_odds: head_sums[head] || 0,
      });
    }
    return result;
  }

  for (let head = 1; head <= 6; head++) {
    const raw = head_sums[head] || 0;
    const p_mkt = raw / totalSum; // 各艇の市場単勝確率
    const synth_odds = p_mkt > 0 ? 1 / p_mkt : null;

    result.push({
      head,
      p_mkt,
      synth_odds,
      raw_sum_inv_odds: raw,
    });
  }

  return result;
}

/**
 * パスから date / pid / race 抜き出し
 * 例: public/odds/v1/2024/0301/01/9R.json
 */
function parseMetaFromPath(filePath, oddsRoot) {
  const rel = path.relative(oddsRoot, filePath); // 2024/0301/01/9R.json
  const parts = rel.split(path.sep);
  if (parts.length < 4) return null;

  const [year, mmdd, pid, raceFile] = parts.slice(-4);
  const raceMatch = raceFile.match(/(\d+)R\.json$/);
  if (!raceMatch) return null;

  const race = Number(raceMatch[1]);
  if (!Number.isInteger(race)) return null;

  const date = `${year}${mmdd}`; // "20240301"

  return { date, pid, race };
}

// ---------- main ----------
async function main() {
  // 出力ディレクトリ作成
  const outDir = path.dirname(OUT_CSV);
  if (outDir && outDir !== "." && !fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const outStream = fs.createWriteStream(OUT_CSV, { encoding: "utf8" });
  outStream.write("date,pid,race,head,p_mkt,synth_odds,raw_sum_inv_odds\n");

  let raceCount = 0;
  let entryCount = 0;

  for await (const file of walkFiles(ODDS_ROOT)) {
    if (!file.endsWith("R.json")) continue;

    const meta = parseMetaFromPath(file, ODDS_ROOT);
    if (!meta) continue;

    let json;
    try {
      const text = await fsp.readFile(file, "utf8");
      json = JSON.parse(text);
    } catch (e) {
      console.warn(`[WARN] JSON parse failed: ${file}: ${e.message}`);
      continue;
    }

    const entries = getTrifectaEntries(json);
    if (!entries.length) continue;

    const heads = calcHeadProbsFrom3T(entries);
    raceCount += 1;
    entryCount += entries.length;

    for (const h of heads) {
      const line = [
        meta.date,
        meta.pid,
        meta.race,
        h.head,
        h.p_mkt.toFixed(6),
        h.synth_odds != null ? h.synth_odds.toFixed(6) : "",
        h.raw_sum_inv_odds.toFixed(8),
      ].join(",");
      outStream.write(line + "\n");
    }
  }

  outStream.end();
  outStream.on("finish", () => {
    console.log("[DONE] races:", raceCount, "3T entries:", entryCount);
    console.log("written:", OUT_CSV);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
