// scripts/fetch-exhibition-direct.js
// 出力: public/exhibition/v1/<date>/<pid>/<race>.json
// 依存: Node.js v18+ (global fetch)
// 主要改良点: 気温セレクタ修正 / 安定板検出強化 / フェッチのタイムアウト&リトライ / 連続アクセスのクールダウン / オート起動の堅牢化

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { load as loadHTML } from "cheerio";

// ========= 基本ユーティリティ =========
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function log(...args) { console.log("[beforeinfo]", ...args); }
function die(msg, code=1){ console.error(msg); process.exit(code); }

function usageAndExit() {
  console.error("Usage: node scripts/fetch-exhibition-direct.js <YYYYMMDD> <pid:01..24 or comma> <race: 1R|1..12|1,3,5R...|1..12,auto|auto> [--skip-existing]");
  process.exit(1);
}

const UA = process.env.UA || "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 10000);
const RETRIES = Number(process.env.RETRIES || 3);
const COOLDOWN_MS = Number(process.env.COOLDOWN_MS || 200);
const AUTO_TRIGGER_MIN = Number(process.env.AUTO_TRIGGER_MIN || 15); // 締切N分前で自動発火

const SKIP_EXISTING = process.argv.includes("--skip-existing");

const argvDate = process.argv[2];
const argvPid  = process.argv[3];
const argvRace = process.argv[4];

const DATE       = process.env.TARGET_DATE  || argvDate  || "";
const PIDS       = (process.env.TARGET_PIDS || argvPid   || "").split(",").map(s=>s.trim()).filter(Boolean);
const RACES_EXPR = process.env.TARGET_RACES || argvRace  || "";

if (!DATE || PIDS.length===0 || !RACES_EXPR) usageAndExit();

const normRaceToken = (tok)=> parseInt(String(tok).replace(/[^0-9]/g,""),10);
function expandRaces(expr){
  if (!expr) return [];
  const lower = String(expr).toLowerCase();
  if (lower === "auto") return ["auto"];
  const parts = String(expr).split(",").map(s=>s.trim()).filter(Boolean);
  const out = new Set();
  for (const p of parts){
    if (p.toLowerCase() === "auto"){ out.add("auto"); continue; }
    const m = p.match(/^(\d+)[Rr]?\.\.(\d+)[Rr]?$/);
    if (m){
      const a = +m[1], b = +m[2]; const [s,e] = a<=b ? [a,b] : [b,a];
      for (let i=s;i<=e;i++){ if (i>=1&&i<=12) out.add(i); }
      continue;
    }
    const n = normRaceToken(p);
    if (!Number.isNaN(n) && n>=1 && n<=12) out.add(n);
  }
  const arr = [...out];
  // autoを含む場合はauto優先（= 締切ベースで拾う）。他の指定は無視せず利用。
  arr.sort((a,b)=> (a==="auto")? -1 : (b==="auto")? 1 : (a-b));
  return arr;
}

const RACES = expandRaces(RACES_EXPR);
if (RACES.length===0) usageAndExit();

function ensureDirSync(dir){ fs.mkdirSync(dir,{recursive:true}); }
async function writeJSON(file, data){ ensureDirSync(path.dirname(file)); await fsp.writeFile(file, JSON.stringify(data,null,2)); }

function toJstDate(dateYYYYMMDD, hhmm){
  return new Date(`${dateYYYYMMDD.slice(0,4)}-${dateYYYYMMDD.slice(4,6)}-${dateYYYYMMDD.slice(6,8)}T${hhmm}:00+09:00`);
}
function tryParseTimeString(s){
  if (!s || typeof s!=="string") return null;
  const m = s.match(/(\d{1,2}):(\d{2})/); if (!m) return null;
  const hh = m[1].padStart(2,"0"), mm=m[2]; return `${hh}:${mm}`;
}

async function loadRaceDeadlineHHMM(date, pid, raceNo){
  // programs どちらかに存在すればOK
  const relPaths = [
    path.join("public","programs","v2",date,pid,`${raceNo}R.json`),
    path.join("public","programs-slim","v2",date,pid,`${raceNo}R.json`),
  ];
  for (const rel of relPaths){
    const abs = path.join(__dirname,"..",rel);
    if (!fs.existsSync(abs)) continue;
    try{
      const j = JSON.parse(await fsp.readFile(abs,"utf8"));
      const candidates = [
        j.deadlineJST,j.closeTimeJST,j.deadline,j.closingTime,j.startTimeJST,j.postTimeJST,
        j.scheduledTimeJST,j.raceCloseJST,j.startAt,j.closeAt,
        j.info?.deadlineJST,j.info?.closeTimeJST,j.meta?.deadlineJST,j.meta?.closeTimeJST
      ].filter(Boolean);
      for (const c of candidates){
        if (typeof c==="string" && c.includes("T") && c.match(/:\d{2}/)){
          const dt = new Date(c); if (!isNaN(dt)){
            const hh=String(dt.getHours()).padStart(2,"0"), mm=String(dt.getMinutes()).padStart(2,"0");
            return `${hh}:${mm}`;
          }
        }
        const hhmm = tryParseTimeString(String(c)); if (hhmm) return hhmm;
      }
      // 生JSONからHH:MMを拾う荒技フォールバック
      const raw = JSON.stringify(j); const m = raw.match(/(\d{1,2}):(\d{2})/);
      if (m) return `${m[1].padStart(2,"0")}:${m[2]}`;
    }catch{}
  }
  return null;
}

async function pickRacesAuto(date, pid){
  const nowMin = Math.floor(Date.now()/60000);
  const out = [];
  for (let r=1;r<=12;r++){
    const hhmm = await loadRaceDeadlineHHMM(date,pid,r); if (!hhmm) continue;
    const triggerMin = Math.floor((toJstDate(date,hhmm).getTime() - AUTO_TRIGGER_MIN*60000)/60000);
    if (nowMin >= triggerMin) out.push(r);
  }
  return out;
}

async function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

async function fetchWithRetry(url, {headers={}, retries=RETRIES, timeoutMs=TIMEOUT_MS}={}){
  let lastErr;
  for (let i=0;i<retries;i++){
    const ctrl = new AbortController();
    const t = setTimeout(()=>ctrl.abort(), timeoutMs);
    try{
      const res = await fetch(url, { headers, signal: ctrl.signal });
      clearTimeout(t);
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
      return await res.text();
    }catch(err){
      clearTimeout(t);
      lastErr = err;
      const backoff = 400*(i+1);
      await sleep(backoff);
    }
  }
  throw lastErr || new Error("fetch failed");
}

// ★ 風向クラス番号 → 日本語方位（1=北 … 13=西 … 16=北北西）
const WIND_DIR_MAP = {
  1:"北", 2:"北北東", 3:"北東", 4:"東北東", 5:"東",
  6:"東南東", 7:"南東", 8:"南南東", 9:"南", 10:"南南西",
  11:"南西", 12:"西南西", 13:"西", 14:"西北西", 15:"北西", 16:"北北西"
};

function parseBeforeinfo(html, {date,pid,raceNo,url}){
  const $ = loadHTML(html);

  // --- ST（右側の図） ---
  const stByLane = {};
  $("div.table1_boatImage1").each((_, el) => {
    const laneText = $(el).find(".table1_boatImage1Number,[class*='table1_boatImage1Number']").text().trim();
    const timeText = $(el).find(".table1_boatImage1Time,[class*='table1_boatImage1Time']").text().trim();
    const lane = parseInt(laneText,10);
    if (lane>=1 && lane<=6) stByLane[lane] = timeText || "";
  });

  // --- 水面気象（右側） ---
  const wRoot = $(".weather1");
  const weatherText = wRoot.find(".weather1_bodyUnit.is-weather .weather1_bodyUnitLabelTitle").text().trim() || null;

  // 気温（正: is-temperature）
  const tempTxt = wRoot.find(".weather1_bodyUnit.is-temperature .weather1_bodyUnitLabelData").first().text().trim();
  const temperature = tempTxt ? parseFloat(tempTxt.replace(/[^\d.]/g,"")) : null;

  // 風速
  const windTxt = wRoot.find(".weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData").text().trim();
  const windSpeed = windTxt ? parseFloat(windTxt.replace(/[^\d.]/g,"")) : null;

  // 風向（クラス is-windNN / 画像alt / タイトル など複数フォールバック）
  let windDirection = null;
  const dirEl = wRoot.find(".weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage");
  const dirClass = (dirEl.attr("class") || "");
  const mDir = dirClass.match(/is-wind(\d{1,2})/);
  if (mDir) {
    const key = parseInt(mDir[1],10);
    windDirection = WIND_DIR_MAP[key] || null;
  } else {
    const alt = dirEl.attr("alt") || dirEl.attr("title") || "";
    // altに「北東」などが入っていたらそれを採用
    if (alt) windDirection = alt.replace(/\s+/g,"").trim() || null;
  }

  // 水温
  const waterTxt = wRoot.find(".weather1_bodyUnit.is-waterTemperature .weather1_bodyUnitLabelData").text().trim();
  const waterTemperature = waterTxt ? parseFloat(waterTxt.replace(/[^\d.]/g,"")) : null;

  // 波高（cm → m）
  const waveTxt = wRoot.find(".weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData").text().trim();
  const waveHeight = waveTxt ? (parseFloat(waveTxt.replace(/[^\d.]/g,""))/100) : null;

  // 安定板使用（見出しやラベル群に "安定板" を含むかを広く検出）
  const stabilizer = $(
    ".title16_titleLabels__add2020 .label1, .title16_titleLabels__add2020 .label2, .weather1, .title16"
  ).filter((_, el) => $(el).text().includes("安定板")).length > 0;

  // --- 直前情報テーブル（左側） ---
  // boatrace公式は tbodyが艇番順に並ぶ構造（is-w748）
  const entries = [];
  const tbodies = $('table.is-w748 tbody');
  tbodies.each((i, tbody) => {
    const lane = i + 1;
    const $tb = $(tbody);

    let number = "", name = "";
    $tb.find('a[href*="toban="]').each((_, a) => {
      const href = $(a).attr("href") || "";
      const m = href.match(/toban=(\d{4})/); if (m) number = m[1];
      const t = $(a).text().replace(/\s+/g," ").trim(); if (t) name = t;
    });

    // 重量 / 展示タイム / チルトのセル群を抽出（位置ズレ耐性）
    const firstRowTds = $tb.find("tr").first().find("td").toArray();
    const texts = firstRowTds.map(td => ($(td).text()||"").replace(/\s+/g,"").trim());
    let weight="", tenjiTime="", tilt="";
    const kgIdx = texts.findIndex(t=>/kg$/i.test(t));
    if (kgIdx!==-1){
      weight = texts[kgIdx] || "";
      tenjiTime = texts[kgIdx+1] || "";
      tilt = texts[kgIdx+2] || "";
    } else {
      // 位置検出に失敗した場合のフォールバック（全テキストから類推）
      const w = texts.find(t=>/kg$/i.test(t)); if (w) weight = w;
      const tt = texts.find(t=>/^\d+\.\d+$/i.test(t)); if (tt) tenjiTime = tt;
      const tl = texts.find(t=>/^-?\d+(\.\d+)?$/i.test(t) && t.includes(".")) || texts.find(t=>/^-?0(\.0)?$/.test(t));
      if (tl) tilt = tl;
    }

    const st = stByLane[lane] || "";
    const stFlag = st.startsWith("F") ? "F" : "";

    entries.push({ lane, number, name, weight, tenjiTime, tilt, st, stFlag });
  });

  return {
    date, pid, race: `${raceNo}R`, source: url, mode: "beforeinfo",
    generatedAt: new Date().toISOString(),
    weather: {
      weather: weatherText ? weatherText.replace(/(り|のち.*)?$/,"") : null, // 例: "曇り"→"曇"
      temperature,
      windSpeed,
      windDirection,
      waterTemperature,
      waveHeight,
      stabilizer
    },
    entries
  };
}

async function fetchBeforeinfo({date,pid,raceNo}){
  const url = `https://www.boatrace.jp/owpc/pc/race/beforeinfo?hd=${date}&jcd=${pid}&rno=${raceNo}`;
  log("GET", url);
  const html = await fetchWithRetry(url, { headers: { "user-agent": UA, "accept-language":"ja,en;q=0.8" }});
  return { url, html };
}

async function main(){
  log(`start: date=${DATE} pids=${PIDS.join(",")} races=${RACES_EXPR} skip=${SKIP_EXISTING}`);
  for (const pid of PIDS){
    let targetRaces = [];

    // auto を含む => autoで拾った集合 ∪ 明示指定レース
    const hasAuto = RACES.includes("auto");
    if (hasAuto){
      const autoList = await pickRacesAuto(DATE, pid);
      const explicit = RACES.filter(r=>r!=="auto");
      const set = new Set([ ...autoList, ...explicit ]);
      targetRaces = [...set].sort((a,b)=>a-b);
      log(`auto-picked races (${pid}): ${autoList.join(", ") || "(none)"}; final=${targetRaces.join(", ") || "(none)"}`);
      if (targetRaces.length===0) continue;
    } else {
      targetRaces = RACES.slice().sort((a,b)=>a-b);
    }

    for (const raceNo of targetRaces){
      const outPath = path.join(__dirname,"..","public","exhibition","v1",DATE,pid,`${raceNo}R.json`);
      if (SKIP_EXISTING && fs.existsSync(outPath)){
        log("skip existing:", path.relative(process.cwd(), outPath));
        continue;
      }
      try{
        const { url, html } = await fetchBeforeinfo({date:DATE,pid,raceNo});
        const data = parseBeforeinfo(html,{date:DATE,pid,raceNo,url});
        if (!data.entries || data.entries.length===0){
          log(`no entries -> skip save: ${DATE}/${pid}/${raceNo}R`);
          continue;
        }
        await writeJSON(outPath, data);
        log("saved:", path.relative(process.cwd(), outPath));
        if (COOLDOWN_MS>0) await sleep(COOLDOWN_MS);
      }catch(err){
        console.error(`Failed: date=${DATE} pid=${pid} race=${raceNo} -> ${String(err?.message || err)}`);
      }
    }
  }
  log("done.");
}

main().catch(e=>{ console.error(e); process.exit(1); });
