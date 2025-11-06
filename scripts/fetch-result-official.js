#!/usr/bin/env node
import fs from "node:fs";
import path from "path";
import { load } from "cheerio";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const z2 = (n) => String(n).padStart(2, "0");
const nowISO = () => new Date().toISOString();
function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }

function toHalf(s = "") {
  return String(s).replace(
    /[０-９Ａ-Ｚａ-ｚ．－−，：’”″′]/g,
    (ch) => {
      const map = { "．": ".", "－": "-", "−": "-", "，": ",", "：": ":", "’": "'", "”": "\"", "″": "″", "′": "′" };
      if (map[ch]) return map[ch];
      return String.fromCharCode(ch.charCodeAt(0) - 0xFEE0);
    }
  );
}
function tx(s) { return toHalf(String(s)).replace(/\s+/g, " ").trim(); }
function num(s) {
  if (s == null) return null;
  const n = Number(String(s).replace(/[^0-9.\-]/g, ""));
  return Number.isFinite(n) ? n : null;
}
function buildOut(hd, jcd, rno) {
  const yyyy = hd.slice(0, 4), mmdd = hd.slice(4);
  return path.join("data", "results", yyyy, mmdd, z2(jcd), `${rno}R.json`);
}

async function fetchHtml(url, { retries = 3, delay = 1000 } = {}) {
  let err;
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, {
        headers: {
          "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
          "accept-language": "ja,en;q=0.9",
        },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    } catch (e) {
      err = e;
      if (i < retries) await sleep(delay * (i + 1));
    }
  }
  throw err;
}

// 時間表記はプライム記号で統一（JSONで \ を出さないため）: 1′52″7
function normalizeTime(raw) {
  if (!raw) return null;
  let s = String(raw)
    .replace(/undefined/g, "″")            // 1'52undefined7 → 1'52″7
    .replace(/[：:]/g, ":")
    .replace(/["”]/g, "″")
    .replace(/[’']/g, "′")
    .replace(/\s+/g, "");
  // 許容: 1:52″7 / 1′52″7 / 1.52.7 / 1'52"7
  const m = s.match(/^(\d)[:.′']?(\d{2})[″"]?(\d)$/);
  if (m) return `${m[1]}′${m[2]}″${m[3]}`;
  return s || null;
}

// 右カラムの気象・決まり手・タイトル
function parseMeta($) {
  const meta = { title: null, decision: null, weather_sky: null, wind_dir: null, wind_speed_m: null, wave_height_cm: null };

  meta.title = tx($(".heading2_titleName").first().text()) || null;

  const dec = tx($("table:contains('決まり手')").first().find("tbody td").first().text());
  if (dec) meta.decision = dec;

  const wx = $(".weather1");
  if (wx.length) {
    const sky = tx(wx.find(".weather1_bodyUnit.is-weather .weather1_bodyUnitLabel").text());
    if (sky) meta.weather_sky = sky.replace(/^(気温|水温|風速|波高)\s*/,"").trim() || null;

    const ws = tx(wx.find(".weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData").text());
    const wsN = ws.match(/([0-9]+(?:\.[0-9]+)?)m/i);
    if (wsN) meta.wind_speed_m = Number(wsN[1]);

    // ⬇ 風向は数値（is-wind13 → 13）
    const dirEl = wx.find(".weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage");
    const cls = (dirEl.attr("class") || "").split(/\s+/).find((c) => /^is-wind\d+/.test(c));
    if (cls) {
      const d = cls.match(/is-wind(\d+)/)?.[1];
      if (d) meta.wind_dir = Number(d);
    }

    const wh = tx(wx.find(".weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData").text());
    const whN = wh.match(/([0-9]+)cm/i);
    if (whN) meta.wave_height_cm = Number(whN[1]);
  }
  return meta;
}

// 結果テーブル（thead 見出しで特定・tbody分割対応）
function parseResults($) {
  const tbl = $("table:has(thead th:contains('着')):has(thead th:contains('枠')):has(thead th:contains('ボートレーサー')):has(thead th:contains('レースタイム'))").first();
  const rows = [];
  if (!tbl.length) return rows;

  tbl.find("tbody tr").each((_, tr) => {
    const $tr = $(tr);
    const tds = $tr.find("td");
    if (tds.length < 4) return;

    const rank = tx($(tds[0]).text()).replace(/[^\d]/g, "") || null;
    const lane = tx($(tds[1]).text()).replace(/[^\d]/g, "") || null;

    const id = tx($(tds[2]).find("span.is-fs12").first().text()).match(/\d{4}/)?.[0] || null;
    const name = tx($(tds[2]).find("span.is-fs18").first().text()) || null;

    const timeRaw = tx($(tds[3]).text());
    const time = normalizeTime(timeRaw);

    // キーは null を出さない（あとで整形）
    const obj = {};
    if (rank) obj.rank = rank;
    if (lane) obj.lane = lane;
    if (id) obj.racer_id = id;
    if (name) obj.racer_name = name;
    if (time) obj.time = time;
    // st/course は後で埋める
    rows.push(obj);
  });
  return rows;
}

// スタート情報：ブロックの出現順＝ コース 1..6
function parseStart($) {
  const startBlocks = $(".table1_boatImage1");
  const out = [];
  startBlocks.each((i, el) => {
    const $el = $(el);
    const lane = tx($el.find(".table1_boatImage1Number").first().text()).replace(/[^\d]/g, "");
    let stTxt = tx($el.find(".table1_boatImage1TimeInner").first().text());
    const stM = stTxt.match(/(\d?\.\d{2})/);
    if (/^[1-6]$/.test(lane) && stM) {
      out.push({ lane: Number(lane), st: Number(stM[1]), course: i + 1 }); // ここで course = 出現順
    }
  });
  // lane 重複除去（最初の出現を優先）
  const seen = new Set();
  return out.filter(r => (seen.has(r.lane) ? false : seen.add(r.lane)));
}

// 配当
function parsePayouts($) {
  const store = { trifecta: null, trio: null, exacta: null, quinella: null, wide: [], win: null, place: [] };
  const blocks = $("table:has(thead th:contains('勝式'))");
  if (!blocks.length) return store;

  blocks.find("tbody").each((_, tb) => {
    const $tb = $(tb); const trs = $tb.find("tr");
    if (!trs.length) return;
    const t0 = $(trs[0]).find("td");
    if (!t0.length) return;
    const kind = tx($(t0[0]).text());
    const key =
      /3連単/.test(kind) ? "trifecta" :
      /3連複/.test(kind) ? "trio" :
      /2連単/.test(kind) ? "exacta" :
      /2連複/.test(kind) ? "quinella" :
      /拡連複/.test(kind) ? "wide" :
      /単勝/.test(kind) ? "win" :
      /複勝/.test(kind) ? "place" : null;
    if (!key) return;

    const combo = getCombo($(trs[0]));
    const amount = getAmount($(trs[0]));
    const pop = getPopularity($(trs[0]));
    if (!combo || amount == null) return;
    const rec = { combo, amount, ...(pop != null ? { popularity: pop } : {}) };

    if (key === "wide" || key === "place") store[key].push(rec);
    else if (key === "win") store.win = rec;
    else store[key] = rec;
  });

  return store;

  function getCombo($tr) {
    const row = $tr.find(".numberSet1_row").first();
    if (!row.length) return tx($tr.find("td").eq(1).text()) || null;
    const nums = row.find(".numberSet1_number").map((i, el) => tx($(el).text())).get().filter(Boolean);
    const symText = row.text();
    const sym = symText.includes("=") ? "=" : "-";
    if (nums.length === 3) return `${nums[0]}${sym}${nums[1]}${sym}${nums[2]}`;
    if (nums.length === 2) return `${nums[0]}${sym}${nums[1]}`;
    if (nums.length === 1) return nums[0];
    return null;
  }
  function getAmount($tr) { return num(tx($tr.find("td").eq(2).text())); }
  function getPopularity($tr) {
    const v = num(tx($tr.find("td").eq(3).text()));
    return Number.isFinite(v) ? v : null;
  }
}

function parseRefunds($) {
  const hasRefundHead = $("table:has(thead th:contains('返還'))").length > 0;
  if (!hasRefundHead) return [];
  // 詳細空のことが多いので、空配列で返す
  return [];
}

async function main() {
  const [,, hd, jcd, rno] = process.argv;
  if (!hd || !jcd || !rno) {
    console.error("usage: node scripts/fetch-result-official.js YYYYMMDD JCD RNO");
    process.exit(1);
  }

  const outPath = buildOut(hd, jcd, rno);
  if (process.env.SKIP_EXISTING === "true" && fs.existsSync(outPath)) {
    console.log(`[skip] ${outPath}`);
    return;
  }

  const url = `https://www.boatrace.jp/owpc/pc/race/raceresult?rno=${Number(rno)}&jcd=${z2(jcd)}&hd=${hd}`;
  const html = await fetchHtml(url, { retries: 3, delay: 1200 });
  await sleep(Number(process.env.FETCH_DELAY_MS || "900"));
  const $ = load(html);

  const meta0 = parseMeta($);
  const resultsRaw = parseResults($);
  const startList = parseStart($);
  const refunds = parseRefunds($);
  const payouts = parsePayouts($);

  // ST / course を補完（スタート情報の順＝コース）
  const stMap = new Map(startList.map(o => [String(o.lane), o.st]));
  const courseMap = new Map(startList.map(o => [String(o.lane), o.course]));

  const results = resultsRaw.map((r) => {
    const out = { ...r };
    const laneKey = r.lane ? String(r.lane) : null;

    // st は文字列（小数第2位固定）
    if (laneKey && stMap.has(laneKey)) out.st = Number(stMap.get(laneKey)).toFixed(2);
    if (laneKey && courseMap.has(laneKey)) out.course = Number(courseMap.get(laneKey));

    // 要望：start_type / note は出力しない（存在しても削除）
    delete out.start_type;
    delete out.note;

    return out;
  });

  // ペイロード作成（null は極力出さない）
  const meta = {};
  meta.date = hd;
  meta.jcd = z2(jcd);
  meta.rno = Number(rno);
  if (meta0.title) meta.title = meta0.title;
  if (meta0.decision) meta.decision = meta0.decision;
  if (meta0.weather_sky != null) meta.weather_sky = meta0.weather_sky;
  if (meta0.wind_dir != null) meta.wind_dir = meta0.wind_dir;             // 数値
  if (meta0.wind_speed_m != null) meta.wind_speed_m = meta0.wind_speed_m;
  if (meta0.wave_height_cm != null) meta.wave_height_cm = meta0.wave_height_cm;

  const payload = {
    meta,
    results,
    start: startList.map(({ lane, st }) => ({ lane, st })), // start は lane/st のみ
    refunds,
    payouts,
    source_url: url,
    generated_at: nowISO(),
  };

  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2), "utf-8");
  console.log("[ok]", outPath);
}

main().catch((e) => { console.error("[error]", e?.message || e); process.exit(1); });
