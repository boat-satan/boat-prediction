#!/usr/bin/env node
// 公式サイトの 3連単オッズをスクレイピングして保存（年/日付ディレクトリ制）
// 出力: public/odds/v1/<YYYY>/<MMDD>/<pid>/<race>R.json
//
// 使い方:
//   node scripts/fetch-odds-official-3t.js <YYYYMMDD> <pid:01..24> <race:1..12>
//   環境変数: TARGET_DATE / TARGET_PID / TARGET_RACE / SKIP_EXISTING=1
//
// テーブル構造（公式PC版・3連単オッズ）
//   4行×列バンドル。各バンドル先頭セル(rowspan=4)が2着S。
//   4行に並ぶ“小さな数字セル”が3着T（4つ）。
//   {1..6} − {S} − {T×4} で残った1つが1着F。
//   よって (F,S,T) → odds を全復元できる。

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { load as loadHTML } from "cheerio";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const log  = (...a)=>console.log("[odds3t]", ...a);
const warn = (...a)=>console.warn("[odds3t][warn]", ...a);
const err  = (...a)=>console.error("[odds3t][error]", ...a);

// ---- 引数/環境変数
const DATE = (process.env.TARGET_DATE || process.argv[2] || "").replace(/-/g,"");
const PID  = (process.env.TARGET_PID  || process.argv[3] || "").padStart(2, "0");
const RACE = String(process.env.TARGET_RACE || process.argv[4] || "").replace(/[^0-9]/g,"");
const SKIP_EXISTING = /^(1|true|yes)$/i.test(String(process.env.SKIP_EXISTING||""));

if (!/^\d{8}$/.test(DATE) || !/^\d{2}$/.test(PID) || !/^(?:[1-9]|1[0-2])$/.test(RACE)) {
  err("Usage: node scripts/fetch-odds-official-3t.js <YYYYMMDD> <pid:01..24> <race:1..12>");
  process.exit(1);
}

// ---- 出力パス（年/日付 = YYYY/MMDD）
const YYYY = DATE.slice(0, 4);
const MMDD = DATE.slice(4, 8);
const OUT_DIR = path.join(__dirname, "..", "public", "odds", "v1", YYYY, MMDD, PID);
const OUT_FILE = `${RACE}R.json`;
const OUT_PATH = path.join(OUT_DIR, OUT_FILE);

function officialOdds3tUrl({date, pid, race}) {
  return `https://www.boatrace.jp/owpc/pc/race/odds3t?rno=${race}&jcd=${pid}&hd=${date}`;
}

async function fetchText(url) {
  const res = await fetch(url, {
    headers: {
      "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36",
      "accept-language": "ja,en;q=0.8",
      "cache-control": "no-cache"
    }
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  return await res.text();
}

const norm  = (s)=>String(s||"").replace(/\s+/g," ").trim();
const toNum = (s)=> {
  const m = String(s||"").match(/-?\d+(?:\.\d+)?/);
  return m ? Number(m[0]) : NaN;
};
const isFiniteNum = (x)=> Number.isFinite(x) && !Number.isNaN(x);

// ---- 「3連単オッズ」テーブルを特定（見出し→近傍 table 優先、フォールバックあり）
function findOddsTable($) {
  let table = null;

  $("*").each((_, el) => {
    const t = norm($(el).text());
    if (/3連単オッズ/.test(t)) {
      const near = $(el).nextAll("div.table1, section, table").first();
      if (near && near.length) {
        table = near.is("table") ? near : near.find("table").first();
        if (table && table.length) return false;
      }
    }
    return;
  });

  if (!table || !table.length) {
    $("table").each((_, t) => {
      const $t = $(t);
      const head = norm($t.find("thead").text());
      const body = norm($t.find("tbody").text());
      if (/(1|2|3|4|5|6)/.test(head) && /(\d+\.\d|\b\d{2,4}\b)/.test(body)) {
        table = $t;
        return false;
      }
    });
  }
  return table;
}

// ---- 4行×列バンドルを走査して (F,S,T,odds) を復元
function parseTrifecta($, $table) {
  const $tbody = $table.find("tbody").first();
  const rows = $tbody.find("tr").toArray();
  const all = [];

  for (let i = 0; i < rows.length; i += 4) {
    const r0 = $(rows[i]), r1 = $(rows[i+1]), r2 = $(rows[i+2]), r3 = $(rows[i+3]);
    if (!r3 || !r2 || !r1) break;

    // row0: 列バンドルの先頭セル (rowspan=4) を 2着S とみなす
    const bundles = []; // [{S, values:[T×4], odds:[×4]}]
    {
      const cells0 = r0.find("th,td").toArray();
      let k = 0;
      while (k < cells0.length) {
        const $c = $(cells0[k]);
        let S = null;

        if ($c.attr("rowspan") === "4") {
          S = toNum($c.text());
          k++;
        } else {
          k++;
          continue; // レイアウト変形は安全側にスキップ
        }

        const v0 = toNum($(cells0[k])?.text()); k++;
        const o0 = toNum($(cells0[k])?.text()); k++;

        if (isFiniteNum(S) && isFiniteNum(v0) && isFiniteNum(o0)) {
          bundles.push({ S, values: [v0], odds: [o0] });
        }
      }
    }

    // row1..row3 : 各バンドルに T/odds を 2セルずつ追加
    const later = [r1, r2, r3];
    for (let ri = 0; ri < later.length; ri++) {
      const cells = later[ri].find("th,td").toArray();
      let k = 0;
      for (let b = 0; b < bundles.length; b++) {
        const v = toNum($(cells[k++]).text());
        const o = toNum($(cells[k++]).text());
        bundles[b].values.push(v);
        bundles[b].odds.push(o);
      }
    }

    // 各バンドル → (F,S,T,odds) 展開
    for (const b of bundles) {
      const S = b.S;
      const Ts = b.values;   // 3着候補×4
      const Os = b.odds;     // オッズ×4
      const thirdSet = new Set(Ts);

      // {1..6} − {S} − {T×4} → 残り1つが 1着F
      const remain = [1,2,3,4,5,6].filter(n => n !== S && !thirdSet.has(n));
      if (remain.length !== 1) continue; // 欠場や崩れた列は捨てる
      const F = remain[0];

      for (let j = 0; j < Ts.length; j++) {
        const T = Ts[j];
        const odds = Os[j];
        if (isFiniteNum(T) && isFiniteNum(odds)) {
          all.push({ combo: `${F}-${S}-${T}`, F, S, T, odds });
        }
      }
    }
  }

  // 重複解消（後勝ち）＋オッズ昇順
  const map = new Map();
  for (const e of all) {
    const prev = map.get(e.combo);
    if (!prev || prev.odds !== e.odds) map.set(e.combo, e);
  }
  const list = [...map.values()].sort((a,b)=> a.odds - b.odds);

  // 人気順（同値は同順位にせず 1,2,3... の連番）
  list.forEach((e, i) => { e.popularityRank = i + 1; });
  return list;
}

async function main() {
  const url = officialOdds3tUrl({ date: DATE, pid: PID, race: RACE });

  // 既存チェック
  if (SKIP_EXISTING && fs.existsSync(OUT_PATH)) {
    log("skip existing:", path.relative(process.cwd(), OUT_PATH));
    return;
  }

  log("GET", url);
  const html = await fetchText(url).catch(e => {
    throw new Error(`fetch failed: ${e.message}`);
  });

  const $ = loadHTML(html);
  const $table = findOddsTable($);
  if (!$table || !$table.length) {
    throw new Error("odds table not found (layout changed?)");
  }

  const trifecta = parseTrifecta($, $table);
  if (trifecta.length === 0) {
    throw new Error("no trifecta odds parsed");
  }

  const payload = {
    date: DATE,
    pid: PID,
    race: `${RACE}R`,
    source: { odds: url },
    generatedAt: new Date().toISOString(),
    trifecta // [{combo:"F-S-T", F,S,T, odds, popularityRank}, ...] オッズ昇順
  };

  // 出力ディレクトリ作成 & .keep
  const ensureDirs = [
    path.join(__dirname, "..", "public"),
    path.join(__dirname, "..", "public", "odds"),
    path.join(__dirname, "..", "public", "odds", "v1"),
    path.join(__dirname, "..", "public", "odds", "v1", YYYY),
    path.join(__dirname, "..", "public", "odds", "v1", YYYY, MMDD),
    OUT_DIR,
  ];
  for (const dir of ensureDirs) {
    try {
      fs.mkdirSync(dir, { recursive: true });
      const keep = path.join(dir, ".keep");
      if (!fs.existsSync(keep)) fs.writeFileSync(keep, "");
    } catch (e) {
      warn("mkdir/.keep failed:", dir, e.message);
    }
  }

  await fsp.writeFile(OUT_PATH, JSON.stringify(payload, null, 2), "utf8");
  log("saved:", path.relative(process.cwd(), OUT_PATH));
}

main().catch(e => { err(e); process.exit(1); });
