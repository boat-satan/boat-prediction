#!/usr/bin/env node
/**
 * BOATRACE 公式 結果スクレイパ (ESM, Node ≥18)
 * usage: node scripts/fetch-result-official.js YYYYMMDD PID RACE
 *   ex) node scripts/fetch-result-official.js 20250812 01 1
 * env:
 *   SKIP_EXISTING=true … 既存JSONがあればスキップ
 *
 * 出力先: data/results/YYYY/MMDD/{pid}/{race}R.json
 * 出力項目: meta(タイトル/決まり手/天候/風向/風速/波高), results(着/艇/ID/選手/ST/進入/タイム/備考),
 *           start(ST一覧), payouts(各勝式), refunds
 */

import fs from 'node:fs';
import path from 'node:path';
import { load } from 'cheerio';

const UA = 'Mozilla/5.0 (compatible; OddsBot/1.0)';

// ---------- utils ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const z2 = (n) => String(n).padStart(2, '0');
const toHalfDigits = (s = '') =>
  String(s).replace(/[０-９]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xFEE0));

const clean = (t) =>
  String(t ?? '')
    .replace(/\u00a0/g, ' ')
    .replace(/undefined/g, '')
    .replace(/\s+/g, ' ')
    .trim();

const normNum = (t) => toHalfDigits(clean(t));

const yenToNumber = (t) => {
  const s = clean(t).replace(/[¥,\s]/g, '');
  const n = parseInt(s, 10);
  return Number.isFinite(n) ? n : null;
};

const timeNormalize = (t) => {
  const s = clean(t)
    .replace(/['’]\s*(\d{2})\s*(\d)/, (_,$1,$2)=>`'${$1}"${$2}`)
    .replace(/(\d)undefined(\d)/g, '$1"$2');
  return s.includes('"') ? s : s.replace(/(\d)$/, '"$1');
};

const numberArrayFromCell = ($cell) =>
  normNum($cell.text()).split(/[^0-9]+/).filter(Boolean).map((n) => String(parseInt(n, 10)));

const popularityFromCell = ($cell) => {
  const n = parseInt(normNum($cell.text()), 10);
  return Number.isFinite(n) ? n : null;
};

const WIND_DIR_MAP = {
  1: '北', 2: '北東', 3: '東', 4: '南東', 5: '南', 6: '南西', 7: '西', 8: '北西',
};

// ---------- main ----------
(async function main() {
  const [, , DATE, PID_IN, RACE_IN] = process.argv;
  if (!DATE || !PID_IN || !RACE_IN) {
    console.error('usage: node scripts/fetch-result-official.js YYYYMMDD PID RACE');
    process.exit(1);
  }
  const pid = z2(parseInt(String(PID_IN).match(/\d+/)?.[0] ?? PID_IN, 10));
  const raceNo = parseInt(String(RACE_IN).match(/\d+/)?.[0] ?? RACE_IN, 10);
  if (!Number.isFinite(raceNo)) {
    console.error('RACE must be number 1..12');
    process.exit(1);
  }

  const url = `https://www.boatrace.jp/owpc/pc/race/raceresult?rno=${raceNo}&jcd=${pid}&hd=${DATE}`;

  // === 出力先（要件に合わせて変更） ===
  const year = DATE.slice(0, 4);
  const md = DATE.slice(4);
  const outDir = path.join('data', 'results', year, md, pid);
  const outFile = path.join(outDir, `${raceNo}R.json`);
  if (process.env.SKIP_EXISTING === 'true' && fs.existsSync(outFile)) {
    console.log(`[skip] exists: ${outFile}`);
    process.exit(0);
  }

  console.log(`[fetch] ${url}`);
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) {
    console.error(`[error] HTTP ${res.status}`);
    process.exit(2);
  }
  const html = await res.text();
  const $ = load(html);

  // ---- メタ: タイトル・決まり手
  const title =
    clean($('h2').first().text()) ||
    clean($('div.title12').first().text()) ||
    null;

  // 決まり手（数値混入を除去）
  const bodyText = clean($.root().text());
  let decision = null;
  {
    const m = bodyText.match(/(逃げ|差し|まくり差し|まくり|抜き|恵まれ|叩き)/);
    decision = m ? m[1] : null;
  }

  // ---- 水面気象（公式のweather1 ボックス優先）
  const weatherBox = $('div.weather1');
  const weather = {
    sky: clean(weatherBox.find('.is-weather .weather1_bodyUnitLabel').text()) || null,
    windSpeedM: parseFloat(clean(weatherBox.find('.is-wind .weather1_bodyUnitLabelData').text()).replace(/[^\d.\-]/g, '')),
    waveCm: parseFloat(clean(weatherBox.find('.is-wave .weather1_bodyUnitLabelData').text()).replace(/[^\d.\-]/g, '')),
    windDir: null,
  };
  // 風向き（コードから→方角名に）
  const cls = weatherBox.find('.is-windDirection .weather1_bodyUnitImage').attr('class') || '';
  const mDir = cls.match(/is-wind(\d+)/);
  const dirCode = mDir ? parseInt(mDir[1], 10) : null;
  if (dirCode && WIND_DIR_MAP[dirCode]) weather.windDir = WIND_DIR_MAP[dirCode];

  // ---- 着順テーブル
  const results = [];
  let $finishTable = $('table:has(th:contains("レースタイム")):has(th:contains("ボートレーサー"))').first();
  if ($finishTable.length === 0) $finishTable = $('table:has(th:contains("着"))').first();

  $finishTable.find('tr').each((_, tr) => {
    const $tds = $(tr).find('td');
    if ($tds.length < 4) return;

    const rankTxt = normNum($tds.eq(0).text());
    const laneTxt = normNum($tds.eq(1).text());

    const rank = rankTxt || null;
    const lane = laneTxt || null;

    const $info = $tds.eq(2);
    const infoText = normNum($info.text());
    const idMatch = infoText.match(/\b\d{4}\b/);
    const racerId = idMatch ? idMatch[0] : null;

    const name =
      clean($info.find('span').last().text()) ||
      clean(infoText.replace(/\b\d{4}\b/, ''));

    const time = timeNormalize($tds.eq(3).text()) || null;

    // 追加でST/進入/備考を拾えるだけ拾う
    let st = null, course = null, note = null;
    const stCell = $tds.eq(4);
    if (stCell && stCell.length) {
      const stTxt = clean(stCell.text());
      const sm = stTxt.match(/([FL]?\s*[-+]?\d*\.\d+)/);
      st = sm ? sm[1].replace(/\s+/g,'') : null;
    }
    const courseCell = $tds.eq(5);
    if (courseCell && courseCell.length) course = normNum(courseCell.text()) || null;

    if (name && name !== 'ボートレーサー') {
      results.push({
        rank, lane, racer_id: racerId, racer_name: name,
        st, course, time, start_type: st?.[0] && /[FL]/.test(st[0]) ? st[0] : null,
        note
      });
    }
  });

  // ---- スタート情報（公式のボート画像行）
  const start = [];
  let startRemark = null;
  $('table:has(th:contains("スタート情報"))').first()
    .find('.table1_boatImage1').each((_, el) => {
      const lane = parseInt(normNum($(el).find('.table1_boatImage1Number').text()), 10);
      const t = clean($(el).find('.table1_boatImage1TimeInner').text());
      const mm = t.match(/([\-+.0-9]+)/);
      const st = mm ? parseFloat(mm[1]) : null;
      const remark = t.replace(mm ? mm[0] : '', '').trim();
      if (lane) start.push({ lane, st });
      if (remark && !startRemark) startRemark = remark;
    });

  // ---- 返還
  let refunds = [];
  {
    const refundText = clean($('table:has(th:contains("返還")) .numberSet1').text());
    if (refundText) refunds = refundText.split(/[^0-9]+/).filter(Boolean).map((n) => parseInt(n, 10)).filter(Number.isFinite);
  }

  // ---- 払戻（勝式）
  const $payTable = $('table:has(th:contains("勝式"))').first();
  const payouts = { trifecta:null, trio:null, exacta:null, quinella:null, wide:[], win:null, place:[] };

  const tbodies = $payTable.find('tbody').toArray();
  const readLine = ($row) => {
    const tds = $row.find('td');
    return {
      combo: numberArrayFromCell($(tds[1])),
      amount: yenToNumber($(tds[2]).text()),
      popularity: popularityFromCell($(tds[3]))
    };
  };

  if (tbodies[0]) {
    const a = readLine($(tbodies[0]).find('tr').eq(0));
    if (a.combo.length === 3) payouts.trifecta = { combo: a.combo.join('-'), amount: a.amount, popularity: a.popularity };
  }
  if (tbodies[1]) {
    const a = readLine($(tbodies[1]).find('tr').eq(0));
    if (a.combo.length === 3) payouts.trio = { combo: a.combo.sort((x,y)=>x-y).join('='), amount: a.amount, popularity: a.popularity };
  }
  if (tbodies[2]) {
    const a = readLine($(tbodies[2]).find('tr').eq(0));
    if (a.combo.length === 2) payouts.exacta = { combo: a.combo.join('-'), amount: a.amount, popularity: a.popularity };
  }
  if (tbodies[3]) {
    const a = readLine($(tbodies[3]).find('tr').eq(0));
    if (a.combo.length === 2) payouts.quinella = { combo: a.combo.sort((x,y)=>x-y).join('='), amount: a.amount, popularity: a.popularity };
  }
  if (tbodies[4]) {
    $(tbodies[4]).find('tr').each((_, tr) => {
      const a = readLine($(tr));
      if (a.combo.length === 2 && a.amount) payouts.wide.push({ combo: a.combo.sort((x,y)=>x-y).join('='), amount: a.amount, popularity: a.popularity });
    });
  }
  if (tbodies[5]) {
    const a = readLine($(tbodies[5]).find('tr').eq(0));
    if (a.combo.length >= 1) payouts.win = { combo: a.combo[0], amount: a.amount };
  }
  if (tbodies[6]) {
    $(tbodies[6]).find('tr').each((_, tr) => {
      const a = readLine($(tr));
      if (a.combo.length >= 1 && a.amount) payouts.place.push({ combo: a.combo[0], amount: a.amount });
    });
  }

  const data = {
    meta: {
      date: DATE, jcd: pid, rno: raceNo,
      title, decision,
      weather_sky: weather.sky || null,
      wind_dir: weather.windDir || null,
      wind_speed_m: Number.isFinite(weather.windSpeedM) ? weather.windSpeedM : null,
      wave_height_cm: Number.isFinite(weather.waveCm) ? weather.waveCm : null
    },
    results,
    start,
    refunds,
    payouts,
    source_url: url,
    generated_at: new Date().toISOString()
  };

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(outFile, JSON.stringify(data, null, 2));
  console.log(`[ok] ${outFile}`);
})();
