#!/usr/bin/env node
/**
 * 期間×場×1..12R を並列取得
 * usage:
 *   node scripts/run-range-official.js YYYYMMDD YYYYMMDD 03,07,12
 * env:
 *   CONCURRENCY=6      … 同時実行上限（推奨 4〜8）
 *   SLEEP_MS=800       … 1リクエスト後の最小スリープ（ミニマムの礼儀）
 *   SKIP_EXISTING=true … 既存JSONスキップ（子プロセスへ継承）
 */
import { spawn } from 'node:child_process';

const [,, START, END, JCDS_STR] = process.argv;
if (!START || !END) {
  console.error('usage: node scripts/run-range-official.js START(YYYYMMDD) END(YYYYMMDD) [JCDS(03,07,...)]');
  process.exit(1);
}
const jcds = (JCDS_STR || '').split(',').map(s => s.trim()).filter(Boolean);

const toDate = (s) => new Date(`${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}T00:00:00Z`);
const fmt = (d) => d.toISOString().slice(0,10).replace(/-/g,'');
const dates = [];
for (let d = toDate(START); d <= toDate(END); d.setUTCDate(d.getUTCDate()+1)) dates.push(fmt(d));

const CONCURRENCY = parseInt(process.env.CONCURRENCY ?? '6', 10);
const SLEEP_MS = parseInt(process.env.SLEEP_MS ?? '800', 10);

const queue = [];
for (const dt of dates) {
  for (const jcd of jcds) {
    for (let r=1; r<=12; r++) queue.push({ dt, jcd, r });
  }
}

// 公式インデックスから開催場自動検出（jcd未指定時）
async function detectJcdsFor(date) {
  const url = `https://www.boatrace.jp/owpc/pc/race/index?hd=${date}`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0 (compatible; OddsBot/1.0)' }});
  if (!res.ok) return [];
  const html = await res.text();
  const m = html.match(/jcd=(\d{2})/g) || [];
  return [...new Set(m.map(x => x.slice(-2)))];
}

(async () => {
  if (jcds.length === 0) {
    for (const dt of dates) {
      const list = await detectJcdsFor(dt);
      for (const jcd of list) for (let r=1;r<=12;r++) queue.push({ dt, jcd, r });
    }
  }

  console.log(`[plan] tasks=${queue.length}, concurrency=${CONCURRENCY}`);

  let running = 0, idx = 0;
  const runNext = async () => {
    if (idx >= queue.length) return;
    if (running >= CONCURRENCY) return;

    const { dt, jcd, r } = queue[idx++];
    running++;
    await new Promise((resolve) => {
      const ps = spawn(process.execPath, ['scripts/fetch-result-official.js', dt, jcd, String(r)], {
        stdio: 'inherit',
        env: { ...process.env } // SKIP_EXISTING 引き継ぎ
      });
      ps.on('close', async () => {
        running--;
        setTimeout(resolve, SLEEP_MS);
      });
    });

    runNext(); // スケジューラ
  };

  const starters = Array.from({ length: CONCURRENCY }, () => runNext());
  await Promise.all(starters);
  console.log('[done] all tasks finished');
})();
