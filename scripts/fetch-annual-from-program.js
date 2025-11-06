#!/usr/bin/env node
/**
 * programの年→前年の各選手「年間成績」を boatrace-db から収集
 * - 対象：コース別成績・コース別決まり手
 * - 3秒インターバル厳守（同時1接続）
 *
 * Usage:
 *   node scripts/fetch-annual-from-program.js <PROGRAM_PATH or YEAR_DIR> [--year 2019] [--outdir data/racers-annual] [--retries 3] [--overwrite]
 *
 * Examples:
 *   # program JSONを1ファイル指定（パスから2019を推定→前年2018を収集）
 *   node scripts/fetch-annual-from-program.js public/programs/v2/2019/0115/04/12R.json
 *
 *   # 年フォルダを指定（配下のprogram*.jsonを全スキャン、年=2019→前年2018を収集）
 *   node scripts/fetch-annual-from-program.js public/programs/v2/2019
 *
 *   # 年を明示（前年=2018で固定）
 *   node scripts/fetch-annual-from-program.js public/programs/v2/2019 --year 2019
 *
 * 出力:
 *   data/racers-annual/2018/4760.json など
 */

import fs from 'node:fs';
import path from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import process from 'node:process';
import { load as cheerioLoad } from 'cheerio';

const args = process.argv.slice(2);
if (args.length < 1) {
  console.error('Usage: node scripts/fetch-annual-from-program.js <PROGRAM_PATH or YEAR_DIR> [--year 2019] [--outdir data/racers-annual] [--retries 3] [--overwrite]');
  process.exit(1);
}

// ---------- opts
const getOpt = (key, def) => {
  const i = args.findIndex(a => a === `--${key}`);
  if (i >= 0 && args[i+1] && !args[i+1].startsWith('--')) return args[i+1];
  return def;
};
const hasFlag = (key) => args.includes(`--${key}`);

const INPUT = args[0];
const EXPL_YEAR = getOpt('year', null);         // e.g. "2019"
const OUTDIR = getOpt('outdir', 'data/racers-annual');
const RETRIES = parseInt(getOpt('retries', '3'), 10);
const OVERWRITE = hasFlag('overwrite');

// ---------- utils
const ensureDir = (p) => fs.mkdirSync(p, { recursive: true });

/** JSON内・任意オブジェクトから登録番号を収集（racer_id / racerId / reg_no / toban を優先） */
function collectRacerIds(obj) {
  const out = new Set();
  const walk = (x) => {
    if (Array.isArray(x)) return x.forEach(walk);
    if (x && typeof x === 'object') {
      for (const [k, v] of Object.entries(x)) {
        if (['racer_id', 'racerId', 'reg_no', 'toban', 'entryId'].includes(k)) {
          const s = String(v ?? '').trim();
          if (/^\d{4,5}$/.test(s)) out.add(s.replace(/^0+/, '') || '0');
        }
        walk(v);
      }
    } else if (typeof x === 'string' || typeof x === 'number') {
      const s = String(x);
      (s.match(/\b(\d{4,5})\b/g) || []).forEach(n => out.add(n.replace(/^0+/, '') || '0'));
    }
  };
  walk(obj);
  return [...out];
}

/** 入力がファイル or ディレクトリかで program JSON リストを返す */
function listProgramJsons(inputPath) {
  const st = fs.statSync(inputPath);
  if (st.isFile()) return [inputPath];
  if (st.isDirectory()) {
    const files = [];
    const walk = (dir) => {
      for (const name of fs.readdirSync(dir)) {
        const p = path.join(dir, name);
        const s = fs.statSync(p);
        if (s.isDirectory()) walk(p);
        else if (s.isFile() && /\.json$/i.test(name)) files.push(p);
      }
    };
    walk(inputPath);
    // program系っぽいのを優先（任意）
    files.sort();
    return files;
  }
  throw new Error('INPUT must be a file or directory');
}

/** パス or 明示年から “前年” を決める */
function inferPrevYear(inputPath, explicitYear) {
  if (explicitYear) {
    const y = parseInt(explicitYear, 10);
    if (!Number.isFinite(y) || y < 1950) throw new Error('--year が不正です');
    return y - 1;
  }
  // パスから 4桁年を拾う（最初にマッチした年を採用）
  const m = inputPath.match(/(?:^|[\\/])(19|20)\d{2}(?=[\\/]|$)/);
  if (!m) throw new Error('年をパスから特定できません。--year で指定してください');
  const y = parseInt(m[0].replace(/[\\/]/g, ''), 10);
  return y - 1;
}

/** 3秒インターバル厳守のフェッチ（同時1本） */
let lastFetchAt = 0;
async function politeFetch(url, options = {}, retries = 3) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    const now = Date.now();
    const wait = Math.max(0, 3000 - (now - lastFetchAt)); // 3秒
    if (wait > 0) await sleep(wait);
    lastFetchAt = Date.now(); // 時刻を先に押さえる（厳しめ）
    try {
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res;
    } catch (e) {
      lastErr = e;
      if (i < retries) {
        // 失敗時も一定のクールダウンを入れて順守
        await sleep(3000);
      }
    }
  }
  throw lastErr;
}

/** テキスト整形 */
const T = (s='') => String(s).replace(/\s+/g, ' ').trim();

/** 見出しテキストに合致する直近の table を拾う */
function findTableByHeading($, headingText) {
  const $h = $(`:header:contains("${headingText}")`).first();
  if ($h.length) {
    // 見出しの次のtable / 同じセクション直下のtableを探索
    let $tbl = $h.nextAll('table').first();
    if ($tbl.length) return $tbl;
    const $sec = $h.closest('section, .section, .content, .panel, .box, div');
    if ($sec.length) {
      $tbl = $sec.find('table').first();
      if ($tbl.length) return $tbl;
    }
  }
  // フォールバック：ページ内のテーブルのうち、見出しに最も近そうなもの
  const candidates = [];
  $('table').each((i, el) => {
    const txt = T($(el).prev().text() || $(el).parent().text());
    candidates.push({ el, score: (txt.includes(headingText) ? 1 : 0) });
  });
  candidates.sort((a, b) => b.score - a.score);
  return candidates.length ? $(candidates[0].el) : null;
}

/** 汎用テーブル→配列のオブジェクト（ヘッダ日本語そのまま） */
function parseTable($, $table) {
  const rows = [];
  if (!$table || !$table.length) return rows;

  const $trs = $table.find('tr');
  if ($trs.length === 0) return rows;

  // ヘッダ行（th優先）
  const $head = $trs.first();
  const headers = $head.find('th,td').map((i, el) => T($(el).text())).get();

  $trs.slice(1).each((i, tr) => {
    const $tds = $(tr).find('td,th');
    if ($tds.length === 0) return;
    const obj = {};
    $tds.each((j, td) => {
      const key = headers[j] || `col${j+1}`;
      obj[key] = T($(td).text());
    });
    rows.push(obj);
  });
  return rows;
}

/** コース(1〜6)をキーにして辞書化（先頭列がコース番号っぽい列を探す） */
function indexByCourse(rows) {
  if (!rows || !rows.length) return {};
  // 先頭キー名候補
  const firstKeys = Object.keys(rows[0] || {});
  const courseKey = firstKeys.find(k => /コース|枠|lane|ｺｰｽ|枠番/i.test(k)) || firstKeys[0];
  const out = {};
  for (const r of rows) {
    const k = String(r[courseKey] || '').match(/\d/);
    const course = k ? k[0] : null;
    if (!course) continue;
    out[course] = r;
  }
  return out;
}

/** 年間成績ページのスクレイプ */
async function fetchAnnual(regno, year) {
  const url = `https://boatrace-db.net/racer/yresult/regno/${regno}/year/${year}/`;
  const html = await (await politeFetch(url, {}, RETRIES)).text();
  const $ = cheerioLoad(html);

  // コース別成績
  const tblStat = findTableByHeading($, 'コース別成績');
  const statRows = parseTable($, tblStat);
  const courseStats = indexByCourse(statRows);

  // コース別決まり手
  const tblKimarite = findTableByHeading($, 'コース別決まり手');
  const kimariteRows = parseTable($, tblKimarite);
  const courseKimarite = indexByCourse(kimariteRows);

  return {
    regno: String(regno),
    year: String(year),
    source_url: url,
    course_stats: courseStats,           // { "1": {...}, ..., "6": {...} }
    course_kimarite: courseKimarite,     // { "1": {...}, ..., "6": {...} }
    fetched_at: new Date().toISOString(),
  };
}

// ---------- main
(async () => {
  const prevYear = inferPrevYear(INPUT, EXPL_YEAR);
  const programFiles = listProgramJsons(INPUT);

  // program群から登録番号をユニーク収集
  const ids = new Set();
  for (const file of programFiles) {
    try {
      const raw = fs.readFileSync(file, 'utf-8');
      const json = JSON.parse(raw);
      collectRacerIds(json).forEach(id => ids.add(id));
    } catch (e) {
      console.warn(`warn: program読み込み失敗 ${file}: ${e.message}`);
    }
  }
  const regnos = [...ids].filter(s => /^\d{4,5}$/.test(s));
  if (regnos.length === 0) {
    console.error('登録番号が見つかりませんでした。programのキー(racer_id等)を確認してください。');
    process.exit(2);
  }

  const outBase = path.join(OUTDIR, String(prevYear));
  ensureDir(outBase);
  console.log(`### programs : ${programFiles.length} files`);
  console.log(`### prevYear : ${prevYear}`);
  console.log(`### racers   : ${regnos.length}人`);
  console.log(`### outDir   : ${outBase}`);
  console.log('（boatrace-dbは3秒インターバル厳守。時間がかかります）\n');

  let ok = 0, skip = 0, ng = 0;
  for (const id of regnos) {
    const outPath = path.join(outBase, `${id}.json`);
    if (!OVERWRITE && fs.existsSync(outPath)) {
      skip++;
      console.log(`[skip] ${id}`);
      continue;
    }
    try {
      const data = await fetchAnnual(id, prevYear);
      fs.writeFileSync(outPath, JSON.stringify(data, null, 2));
      ok++;
      console.log(`[ok]   ${id} -> ${path.relative(process.cwd(), outPath)}`);
    } catch (e) {
      ng++;
      console.warn(`[fail] ${id} -> ${e?.message || e}`);
    }
  }

  console.log(`\nDone. success=${ok} skip=${skip} fail=${ng}`);
})();
