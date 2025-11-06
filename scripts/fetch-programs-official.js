#!/usr/bin/env node
/**
 * BOATRACE公式 出走表スクレイパー (2025構造対応・複数tbody対応)
 * usage: node scripts/fetch-programs-official.js YYYYMMDD PID RACE
 *
 * 出力: public/programs/v1/YYYY/MMDD/{pid}/{race}R.json
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { load } from "cheerio";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ========== 基本ユーティリティ ==========
function log(...args) { console.log("[program]", ...args); }
function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
async function writeJSON(p, data) {
  ensureDir(path.dirname(p));
  await fsp.writeFile(p, JSON.stringify(data, null, 2));
}
function txt($el) { return $el.text().replace(/\s+/g, " ").trim(); }
function num(v) {
  const m = String(v || "").match(/[\d.]+/);
  return m ? (m[0].includes(".") ? parseFloat(m[0]) : parseInt(m[0], 10)) : null;
}
function isCorrupted(html) {
  return /undefinedhttps?:\/\//.test(html) || /\bundefined[a-zA-Z-]+undefined\b/.test(html);
}

// ========== 引数 ==========
const date = process.argv[2];
const pid = process.argv[3];
const raceNo = process.argv[4];
if (!date || !pid || !raceNo) {
  console.error("Usage: node scripts/fetch-programs-official.js YYYYMMDD PID RACE");
  process.exit(1);
}

const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36";

// ========== Fetch関数 ==========
async function fetchHtml(url) {
  const res = await fetch(url, {
    headers: { "User-Agent": UA, "Accept-Language": "ja,en;q=0.8" },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const html = await res.text();
  if (isCorrupted(html)) {
    throw new Error("HTML破損: 'undefined'混入検出。fetch後の文字列処理を確認してください。");
  }
  return html;
}

// ========== HTMLパーサ ==========
function parseProgram(html, { date, pid, raceNo, url }) {
  const $ = load(html);
  const bodyText = $("body").text();

  // レース情報なし
  if (/レース情報はありません|開催情報はありません|該当するレース情報は存在しません/.test(bodyText)) {
    return {
      date, pid, race: `${raceNo}R`, source: url, mode: "program",
      generatedAt: new Date().toISOString(), entries: [],
    };
  }

  const entries = [];
  const $table = $("div.table1.is-tableFixed__3rdadd table");

  if ($table.length === 0) {
    // フォールバック: シンプル構成 (旧is-w748)
    $("table.is-w748 tbody tr").each((i, el) => {
      const tds = $(el).find("td");
      if (tds.length < 10) return;
      entries.push({
        lane: num($(tds[0]).text()) ?? (i + 1),
        number: txt($(tds[1])),
        name: txt($(tds[2])),
        branch: txt($(tds[3])),
        grade: txt($(tds[4])),
        age: num($(tds[5]).text()),
        weight: txt($(tds[6])),
        motor: txt($(tds[7])),
        boat: txt($(tds[8])),
        winRate: txt($(tds[9])),
        localRate: txt($(tds[10])),
        stAvg: txt($(tds[11])),
      });
    });
  } else {
    // 2025年構造: tbodyごとに1枠 (4行構成)
    $table.find("> tbody").each((_, tb) => {
      const rows = $(tb).find("> tr");
      if (rows.length < 4) return;

      const r1 = $(rows[0]).find("td");
      const r2 = $(rows[1]).find("td");
      const r3 = $(rows[2]).find("td");
      const r4 = $(rows[3]).find("td");

      const lane = num($(r1[0]).text());

      const block = $(r1[2]);
      const regGrade = txt(block.find("div").eq(0)); // "4456 / A1"
      const name = txt(block.find("div").eq(1)); // "鎌倉 涼"
      const branchWeight = txt(block.find("div").eq(2)); // "大阪/大阪 36歳/45.0kg"

      const number = (regGrade.match(/\d{4}/) || [])[0] || "";
      const grade = (regGrade.match(/A1|A2|B1|B2/) || [])[0] || "";
      const branch = (branchWeight.split(/\s+/)[0] || "").split("/")[0] || "";
      const weightText = (branchWeight.match(/\d+(?:\.\d+)?kg/) || [])[0] || "";
      const age = num((branchWeight.match(/(\d+)歳/) || [])[1]);

      const national = txt($(r1[4]));
      const local = txt($(r1[5]));
      const motor = txt($(r1[6]));
      const boat = txt($(r1[7]));

      const entryCourses = r2.toArray().map(td => txt($(td))).filter(Boolean);
      const stList = r3.toArray().map(td => txt($(td))).filter(Boolean);
      const results = r4.toArray().map(td => txt($(td))).filter(Boolean);

      entries.push({
        lane,
        number,
        name,
        branch,
        grade,
        age,
        weight: weightText,
        stats: { national, local, motor, boat },
        recent: { entryCourses, stList, results },
      });
    });
  }

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

// ========== メイン ==========
async function main() {
  const url = `https://www.boatrace.jp/owpc/pc/race/racelist?rno=${raceNo}&jcd=${pid}&hd=${date}`;
  log("GET", url);
  try {
    const html = await fetchHtml(url);
    const data = parseProgram(html, { date, pid, raceNo, url });
    const year = date.slice(0, 4);
    const md = date.slice(4);
    const outPath = path.join(__dirname, "..", "public", "programs", "v1", year, md, pid, `${raceNo}R.json`);
    await writeJSON(outPath, data);
    log("saved:", outPath);
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
  }
}

main();
