#!/usr/bin/env node
/**
 * BOATRACE 公式 結果スクレイパ (ESM, Node ≥20)
 * usage: node scripts/fetch-result-official.js YYYYMMDD PID RACE
 *   ex) node scripts/fetch-result-official.js 20250812 04 1
 * env:
 *   SKIP_EXISTING=true  …… 既存JSONがあればスキップ
 *   FETCH_DELAY_MS=1200 …… アクセス間隔 (ms)
 * 出力先: data/results/YYYY/MMDD/{pid}/{race}R.json
 * 出力項目: meta(タイトル/決まり手/天候/風向/風速/波高), results(着/艇/ID/選手/ST/進入/タイム/備考),
 *           start(ST一覧), payouts(各勝式), refunds
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { load } from 'cheerio';

// ---------- utils
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const z2 = (n) => String(n).padStart(2, '0');

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function toHalfDigits(s = '') {
  return String(s).replace(/[０-９．－−]/g, (ch) => {
    const map = { '．': '.', '－': '-', '−': '-' };
    if (map[ch]) return map[ch];
    const code = ch.charCodeAt(0) - 0xFEE0;
    return String.fromCharCode(code);
  });
}

function numOrNull(s) {
  if (!s) return null;
  const n = Number(s.replace(/[^0-9.\-]/g, ''));
  return Number.isFinite(n) ? n : null;
}

function text($el) {
  return toHalfDigits($el.text().trim().replace(/\s+/g, ' '));
}

function buildOutPath(hd, pid, rno) {
  const yyyy = hd.slice(0, 4);
  const mmdd = hd.slice(4);
  return path.join('data', 'results', yyyy, mmdd, z2(pid), `${rno}R.json`);
}

// ---------- fetch with retry
async function fetchHtml(url, { retries = 3, delay = 1000 } = {}) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, {
        headers: {
          'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
          'accept-language': 'ja,en;q=0.9',
        },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    } catch (e) {
      lastErr = e;
      if (i < retries) await sleep(delay * (i + 1));
    }
  }
  throw lastErr;
}

// ---------- parse blocks
function parseMeta($) {
  const meta = {};
  meta.title = text($('.hdg2, .heading2, .result_hd, .raceTitle').first());

  const envBlock = $('.weather1, .weather, .result_table, .result_info, .info').first();
  const envText = toHalfDigits(envBlock.text());

  const mKimarite = envText.match(/決まり手\s*[:：]?\s*([\u3040-\u30FF\u4E00-\u9FFF・\w]+)/);
  const mTenko = envText.match(/天候\s*[:：]?\s*([\u3040-\u30FF\u4E00-\u9FFF・\w]+)/);
  const mKaze = envText.match(/風\s*[:：]?\s*([\u3040-\u30FF\u4E00-\u9FFF・\w]+)(?:\s*(\d+(?:\.\d+)?)m)?/);
  const mNami = envText.match(/波\s*[:：]?\s*(\d+(?:\.\d+)?)m/);

  meta.kimarite = mKimarite?.[1] || null;
  meta.weather = mTenko?.[1] || null;
  meta.wind_dir = mKaze?.[1] || null;
  meta.wind_speed = mKaze?.[2] ? Number(mKaze[2]) : null;
  meta.wave = mNami ? Number(mNami[1]) : null;

  return meta;
}

function parseResults($) {
  const tables = $('table');
  let target;
  tables.each((_, el) => {
    const $t = $(el);
    const head = $t.find('th').map((i, th) => text($(th))).get().join('|');
    if (/着|着順/.test(head) && /艇|ボート|選手/.test(head)) target = $t;
  });
  if (!target) return [];

  const rows = [];
  target.find('tr').each((_, tr) => {
    const $tr = $(tr);
    const cells = $tr.find('td');
    if (!cells.length) return;

    const txts = cells.map((i, td) => text($(td))).get();
    const data = {
      finish: null,  // 1..6
      lane: null,    // 1..6
      racer_id: null, // 4桁
      racer_name: null,
      st: null,      // 0.14 / F.02 / L.01
      course: null,  // 123/456 or single
      time: null,    // 多様式
      note: null,    // 妨/転/落/失 など
    };

    data.finish = txts.find((s) => /^\d$/.test(s)) || null;
    data.lane = txts.find((s) => /^[1-6]$/.test(s)) || null;

    const idHit = txts.find((s) => /\b\d{4}\b/.test(s));
    data.racer_id = idHit ? idHit.match(/\b\d{4}\b/)?.[0] : null;

    const nameCand = [...txts].sort((a, b) => b.length - a.length)[0] || '';
    data.racer_name = nameCand.replace(/\b\d{4}\b.*/, '').replace(/\s+/g, '').slice(0, 20) || null;

    data.st = (txts.find((s) => /^(?:F\.|L\.)?\d?\.\d{2}$/.test(s)) || null);
    data.course = txts.find((s) => /^(?:[1-6]{1,6}(?:\/[1-6]{1,6})?|[1-6])$/.test(s)) || null;
    data.time = (txts.find((s) => /^(?:\d['’:]\d{2}["”]\d|\d?\.?\d{2}\.\d|\d{2}\.\d{2})$/.test(s)) || null);
    data.note = txts.find((s) => /妨|転|落|エ|失|欠|不/.test(s)) || null;

    rows.push(data);
  });

  return rows.filter((r) => r.lane || r.racer_id || r.racer_name);
}

function parseStartList($) {
  const out = [];
  $('table').each((_, el) => {
    const $t = $(el);
    const head = $t.find('th').map((i, th) => text($(th))).get().join('|');
    if (/ST/i.test(head) && /艇|進入/.test(head)) {
      $t.find('tr').each((_, tr) => {
        const $tr = $(tr);
        const tds = $tr.find('td');
        if (!tds.length) return;
        const lane = text(tds.eq(0));
        const stTxt = text(tds.filter((i, td) => /^(?:F\.|L\.)?\d?\.\d{2}$/.test(text($(td)))).first());
        if (/^[1-6]$/.test(lane) && stTxt) out.push({ lane, st: stTxt });
      });
    }
  });
  return out;
}

function parsePayouts($) {
  const results = [];
  $('table').each((_, el) => {
    const $t = $(el);
    const head = $t.find('th').map((i, th) => text($(th))).get().join('|');
    if (/払戻|配当|勝式/.test(head)) {
      $t.find('tr').each((_, tr) => {
        const $tr = $(tr);
        const tds = $tr.find('td');
        if (tds.length < 3) return;
        const kind = text(tds.eq(0));
        const combo = text(tds.eq(1));
        const amount = numOrNull(text(tds.eq(2)));
        let ninki = null;
        if (tds.eq(3).length) {
          const n = numOrNull(text(tds.eq(3)));
          ninki = Number.isFinite(n) ? n : null;
        }
        if (kind && combo && amount != null) results.push({ kind, combo, amount, ninki });
      });
    }
  });
  return results;
}

function parseRefunds($) {
  const out = [];
  const txt = toHalfDigits($('body').text());
  const lines = txt.split(/\n+/).map((s) => s.trim()).filter(Boolean);
  for (const line of lines) {
    if (/返還|不成立|没収/.test(line)) out.push(line);
  }
  return out;
}

// ---------- main
async function main() {
  const [,, hd, pid, rno] = process.argv;
  if (!hd || !pid || !rno) {
    console.error('usage: node scripts/fetch-result-official.js YYYYMMDD PID RACE');
    process.exit(1);
  }

  const outPath = buildOutPath(hd, pid, rno);
  if (process.env.SKIP_EXISTING === 'true' && fs.existsSync(outPath)) {
    console.log(`[skip] exists: ${outPath}`);
    return;
  }

  const url = `https://www.boatrace.jp/owpc/pc/race/raceresult?rno=${Number(rno)}&jcd=${z2(pid)}&hd=${hd}`;
  const delay = Number(process.env.FETCH_DELAY_MS || '1200');

  console.log(`[fetch] ${url}`);
  const html = await fetchHtml(url, { retries: 3, delay: 1200 });
  await sleep(delay);

  const $ = load(html);
  const meta = parseMeta($);
  const results = parseResults($);
  const start = parseStartList($);
  const payouts = parsePayouts($);
  const refunds = parseRefunds($);

  const payload = {
    meta: {
      ...meta,
      pid: z2(pid),
      rno: Number(rno),
      date: hd,
      source: url,
      scraped_at: new Date().toISOString(),
    },
    results,
    start,
    payouts,
    refunds,
  };

  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf-8');
  console.log(`[ok] ${outPath}`);
}

main().catch((e) => {
  console.error('[error]', e?.message || e);
  process.exit(1);
});
