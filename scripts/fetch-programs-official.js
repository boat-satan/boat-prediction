#!/usr/bin/env node
// ===========================================
// BOATRACE 出走表スクレイパ（公式サイト版）
// 出力: public/programs/v1/<YYYY>/<MMDD>/<pid>/<race>R.json
// 依存: Node.js v18+（fetch標準対応） + cheerio
// ===========================================

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { load } from "cheerio";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const UA = process.env.UA || "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 10000);
const RETRIES = Number(process.env.RETRIES || 3);
const COOLDOWN_MS = Number(process.env.COOLDOWN_MS || 200);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchWithRetry(url, retries = RETRIES) {
  let lastErr;
  for (let i = 0; i < retries; i++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(url, {
        headers: { "user-agent": UA, "accept-language": "ja,en;q=0.8" },
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    } catch (err) {
      clearTimeout(timer);
      lastErr = err;
      console.warn(`Retry ${i + 1}/${retries}: ${err.message}`);
      await sleep(300 * (i + 1));
    }
  }
  throw lastErr || new Error("fetch failed");
}

function toNum(s) {
  if (!s) return null;
  const v = parseFloat(String(s).replace(/[^\d.-]/g, ""));
  return Number.isFinite(v) ? v : null;
}

function parseProgram(html, { date, pid, raceNo, url }) {
  const $ = load(html);
  const entries = [];

  $(".table1_boatRacer1Body").each((i, el) => {
    const lane = i + 1;
    const number = $(el).find(".table1_boatRacer1Number").text().trim();
    const name = $(el).find(".table1_boatRacer1Name").text().trim();
    const branch = $(el).find(".table1_boatRacer1Place").text().trim();
    const cls = $(el).find(".table1_boatRacer1Rank").text().trim();
    const age = toNum($(el).find(".table1_boatRacer1Age").text());
    const weightKg = toNum($(el).find(".table1_boatRacer1Weight").text());
    const motorNo = toNum($(el).find(".table1_boatRacer1MotorNo").text());
    const boatNo = toNum($(el).find(".table1_boatRacer1BoatNo").text());
    const motor2Rate = toNum($(el).find(".table1_boatRacer1Motor2Win").text());
    const boat2Rate = toNum($(el).find(".table1_boatRacer1Boat2Win").text());

    entries.push({
      lane,
      number,
      name,
      branch,
      class: cls,
      age,
      weightKg,
      motorNo,
      motor2Rate,
      boatNo,
      boat2Rate,
    });
  });

  $(".table1_boatRacer2Body").each((i, el) => {
    const e = entries[i];
    if (!e) return;
    const tds = $(el)
      .find("td")
      .map((_, td) => $(td).text().trim())
      .get();
    e.winRate = toNum(tds[0]);
    e.twoRate = toNum(tds[1]);
    e.threeRate = toNum(tds[2]);
    e.avgST = toNum(tds[3]);
  });

  return {
    date,
    pid,
    race: `${raceNo}R`,
    source: url,
    mode: "program",
    generatedAt: new Date().toISOString(),
    entries,
  };
}

async function main() {
  const [,, date, pid, raceNo] = process.argv;
  if (!date || !pid || !raceNo) {
    console.error("Usage: node scripts/fetch-programs-official.js <YYYYMMDD> <pid> <race>");
    process.exit(1);
  }

  const url = `https://www.boatrace.jp/owpc/pc/race/racelist?rno=${parseInt(raceNo)}&jcd=${pid}&hd=${date}`;
  console.log("[program]", "GET", url);
  const html = await fetchWithRetry(url);
  const data = parseProgram(html, { date, pid, raceNo, url });

  // ==== 出力ディレクトリ: v1/YYYY/MMDD/pid/race.json ====
  const year = date.slice(0, 4);
  const mmdd = date.slice(4, 8);
  const outPath = path.join(
    __dirname,
    "..",
    "public",
    "programs",
    "v1",
    year,
    mmdd,
    pid,
    `${raceNo}R.json`
  );

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await fsp.writeFile(outPath, JSON.stringify(data, null, 2));
  console.log("saved:", path.relative(process.cwd(), outPath));

  if (COOLDOWN_MS > 0) await sleep(COOLDOWN_MS);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
