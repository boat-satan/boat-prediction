#!/usr/bin/env node
/**
 * BOATRACE 公式 出走表スクレイパー (2025対応)
 * usage: node scripts/fetch-programs-official.js YYYYMMDD PID RACE
 * ex)    node scripts/fetch-programs-official.js 20250101 01 1
 *
 * 出力先: public/programs/v1/YYYY/MMDD/{pid}/{race}R.json
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { load } from "cheerio";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------- util ----------
function log(...a) { console.log("[program]", ...a); }
function ensureDirSync(p) { fs.mkdirSync(p, { recursive: true }); }
async function writeJSON(p, d) {
  ensureDirSync(path.dirname(p));
  await fsp.writeFile(p, JSON.stringify(d, null, 2));
}

// ---------- args ----------
const date = process.argv[2];
const pid = process.argv[3];
const raceNo = process.argv[4];
if (!date || !pid || !raceNo) {
  console.error("Usage: node scripts/fetch-programs-official.js YYYYMMDD PID RACE");
  process.exit(1);
}

const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36";

// ---------- main ----------
async function fetchHtml(url) {
  const res = await fetch(url, { headers: { "User-Agent": UA, "Accept-Language": "ja,en;q=0.8" } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.text();
}

function parseProgram(html, ctx) {
  const { date, pid, raceNo, url } = ctx;
  const $ = load(html);

  const textAll = $("body").text();
  if (
    textAll.includes("レース情報はありません") ||
    textAll.includes("開催情報はありません") ||
    textAll.includes("該当するレース情報は存在しません")
  ) {
    log(`skip: no info ${date}/${pid}/${raceNo}`);
    return { date, pid, race: `${raceNo}R`, source: url, mode: "program", generatedAt: new Date().toISOString(), entries: [] };
  }

  const entries = [];

  // 両対応: .table1_boatRacer1Body と .table1_boatRacer1 のどちらでも拾う
  const $rows = $(".table1_boatRacer1Body, .table1_boatRacer1");
  if ($rows.length === 0) {
    log(`warn: no entry rows found ${date}/${pid}/${raceNo}`);
  }

  $rows.each((i, el) => {
    const lane = i + 1;
    const number = $(el).find(".table1_boatRacer1Number").text().trim() || "";
    const name = $(el).find(".table1_boatRacer1Name").text().trim() || "";
    const branch = $(el).find(".table1_boatRacer1Branch").text().trim() || "";
    const classText = $(el).find(".table1_boatRacer1Rank").text().trim() || "";
    const ageText = $(el).find(".table1_boatRacer1Age").text().trim() || "";
    const weightText = $(el).find(".table1_boatRacer1Weight").text().trim() || "";
    const motor = $(el).find(".table1_boatRacer1MotorNo").text().trim() || "";
    const boat = $(el).find(".table1_boatRacer1BoatNo").text().trim() || "";
    const winRate = $(el).find(".table1_boatRacer1ST").text().trim() || "";
    const twoRate = $(el).find(".table1_boatRacer1ST2").text().trim() || "";

    entries.push({
      lane,
      number,
      name,
      branch,
      class: classText,
      age: ageText,
      weight: weightText,
      motor,
      boat,
      winRate,
      twoRate
    });
  });

  return {
    date,
    pid,
    race: `${raceNo}R`,
    source: url,
    mode: "program",
    generatedAt: new Date().toISOString(),
    entries
  };
}

async function main() {
  const url = `https://www.boatrace.jp/owpc/pc/race/racelist?rno=${raceNo}&jcd=${pid}&hd=${date}`;
  log("GET", url);

  try {
    const html = await fetchHtml(url);
    const data = parseProgram(html, { date, pid, raceNo, url });

    // 年/月日階層で保存
    const year = date.slice(0, 4);
    const md = date.slice(4);
    const outPath = path.join(__dirname, "..", "public", "programs", "v1", year, md, pid, `${raceNo}R.json`);
    await writeJSON(outPath, data);
    log("saved:", outPath);
  } catch (err) {
    console.error("ERROR:", err.message);
  }
}

main();
