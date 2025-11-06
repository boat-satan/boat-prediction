#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { load } from "cheerio";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const z2 = (n) => String(n).padStart(2, "0");
const nowISO = () => new Date().toISOString();

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function toHalf(s=""){ return String(s).replace(/[０-９Ａ-Ｚａ-ｚ．－−，：’”″′]/g, ch=>{
  const map={ "．":".","－":"-","−":"-","，":",","：":":","’":"'","”":"\"","″":"\"","′":"'" };
  if(map[ch]) return map[ch];
  return String.fromCharCode(ch.charCodeAt(0)-0xFEE0);
});}
function tx(s){ return toHalf(String(s)).replace(/\s+/g," ").trim(); }
function num(s){ if(s==null) return null; const n=Number(String(s).replace(/[^0-9.\-]/g,"")); return Number.isFinite(n)?n:null; }
function buildOut(hd,jcd,rno){ const yyyy=hd.slice(0,4),mmdd=hd.slice(4); return path.join("data","results",yyyy,mmdd,z2(jcd),`${rno}R.json`); }

async function fetchHtml(url,{retries=3,delay=1000}={}) {
  let err;
  for(let i=0;i<=retries;i++){
    try{
      const res=await fetch(url,{headers:{
        "user-agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
        "accept-language":"ja,en;q=0.9",
      }});
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    }catch(e){ err=e; if(i<retries) await sleep(delay*(i+1)); }
  }
  throw err;
}

// --- HTML内 “undefined” ノイズに耐えるタイム正規化
function normalizeTime(raw){
  if(!raw) return null;
  let s = String(raw).replace(/undefined/g, "\"");             // 1'52undefined7 → 1'52"7
  s = s.replace(/[：:]/g,":").replace(/[″”"]/g,'"').replace(/[′’']/g,"'");
  // 1:52"7 / 1'52"7 / 1.52.7 → 1'52"7
  const m = s.match(/^(\d)[:.'](\d{2})[".](\d)$/);
  if(m) return `${m[1]}'${m[2]}"${m[3]}`;
  return s.trim() || null;
}

// --- 方位アイコン → 風向（簡易マップ）
const WIND_MAP = {
  "is-wind1":"北", "is-wind2":"北北東", "is-wind3":"北東", "is-wind4":"東北東",
  "is-wind5":"東", "is-wind6":"東南東", "is-wind7":"南東", "is-wind8":"南南東",
  "is-wind9":"南", "is-wind10":"南南西","is-wind11":"南西","is-wind12":"西南西",
  "is-wind13":"西", "is-wind14":"西北西","is-wind15":"北西","is-wind16":"北北西"
};

// --- Meta（タイトル/決まり手/天候/風/波）: 右カラムから拾う
function parseMeta($){
  const meta = { title:null, decision:null, weather_sky:null, wind_dir:null, wind_speed_m:null, wave_height_cm:null };

  // タイトル（場名＋開催名の見出し近辺）
  meta.title = tx($(".heading2_titleName").first().text()) || null;

  // 決まり手
  const dec = tx($("table:contains('決まり手')").first().find("tbody td").first().text());
  if(dec) meta.decision = dec;

  // 気象（右カラム）
  const wx = $(".weather1");
  if(wx.length){
    // 天候テキスト（雨/晴など）
    const sky = tx(wx.find(".weather1_bodyUnit.is-weather .weather1_bodyUnitLabel").text());
    if(sky) meta.weather_sky = sky.replace(/^(気温|水温|風速|波高)\s*/,"").trim();

    // 風速
    const ws = tx(wx.find(".weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData").text());
    const wsN = ws.match(/([0-9]+(?:\.[0-9]+)?)m/i);
    if(wsN) meta.wind_speed_m = Number(wsN[1]);

    // 風向（アイコンclass）
    const dirEl = wx.find(".weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage");
    const cls = (dirEl.attr("class")||"").split(/\s+/).find(c=>/^is-wind\d+/.test(c));
    if(cls && WIND_MAP[cls]) meta.wind_dir = WIND_MAP[cls];

    // 波高
    const wh = tx(wx.find(".weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData").text());
    const whN = wh.match(/([0-9]+)cm/i);
    if(whN) meta.wave_height_cm = Number(whN[1]);
  }
  return meta;
}

// --- 結果テーブル（thead固定: 着/枠/ボートレーサー/レースタイム, 複数tbody対応）
function parseResults($){
  const tbl = $("table:has(thead th:contains('着')):has(thead th:contains('枠')):has(thead th:contains('ボートレーサー')):has(thead th:contains('レースタイム'))").first();
  const rows = [];
  if(!tbl.length) return rows;

  tbl.find("tbody tr").each((_,tr)=>{
    const $tr=$(tr);
    const tds=$tr.find("td");
    if(tds.length<4) return;

    const rank = tx($(tds[0]).text()).replace(/[^\d]/g,"") || null;
    const lane = tx($(tds[1]).text()).replace(/[^\d]/g,"") || null;

    // ID と 名前は 3列目に二つの span
    const id = tx($(tds[2]).find("span.is-fs12").first().text()).match(/\d{4}/)?.[0] || null;
    const name = tx($(tds[2]).find("span.is-fs18").first().text()) || null;

    const timeRaw = tx($(tds[3]).text());
    const time = normalizeTime(timeRaw);

    if(rank || lane || id || name){
      rows.push({ rank, lane, racer_id:id, racer_name:name, st:null, course:null, time: time || null, start_type:null, note:null });
    }
  });
  return rows;
}

// --- スタート情報（.table1_boatImage1 から lane と ST、勝者行の語を start_type に）
function parseStart($){
  const out=[]; let winnerType=null;
  $(".table1_boatImage1").each((i,el)=>{
    const $el=$(el);
    const lane = tx($el.find(".table1_boatImage1Number").first().text()).replace(/[^\d]/g,"");
    let stTxt = tx($el.find(".table1_boatImage1TimeInner").first().text());
    // `.12   抜き` のようにSTと語が並ぶので分離
    const stM = stTxt.match(/(\d?\.\d{2})/);
    const typeM = stTxt.replace(stM?.[1]??"","").trim();
    if(/^[1-6]$/.test(lane) && stM){
      out.push({ lane: Number(lane), st: Number(stM[1]) });
      if(winnerType==null && typeM) winnerType = typeM;
    }
  });
  return { startList: dedupLane(out), winnerType };
}
function dedupLane(list){ const m=new Map(); for(const r of list){ if(!m.has(r.lane)) m.set(r.lane,r); } return [...m.values()]; }

// --- 配当：勝式テーブルをDOMで厳密に読む（数字バッジを結合）
function parsePayouts($){
  const store = { trifecta:null, trio:null, exacta:null, quinella:null, wide:[], win:null, place:[] };
  const blocks = $("table:has(thead th:contains('勝式'))");
  if(!blocks.length) return store;

  // tbodyごとに「勝式 行」を読む（行が2つに分かれる構造に対応）
  blocks.find("tbody").each((_,tb)=>{
    const $tb=$(tb); const trs=$tb.find("tr");
    if(!trs.length) return;
    const t0=$(trs[0]).find("td");
    if(!t0.length) return;
    const kind = tx($(t0[0]).text()); // 「3連単」など（rowspanで次行に持ち越しあり）
    const key = (/3連単/.test(kind)?"trifecta":/3連複/.test(kind)?"trio":/2連単/.test(kind)?"exacta":
                 /2連複/.test(kind)?"quinella":/拡連複/.test(kind)?"wide":/単勝/.test(kind)?"win":
                 /複勝/.test(kind)?"place":null);
    if(!key) return;

    // 組番は数字バッジで取得
    const combo = getCombo($(trs[0]));
    const amount = getAmount($(trs[0]));
    const pop = getPopularity($(trs[0]));

    if(!combo || amount==null) return;
    const rec = { combo, amount, ...(pop!=null?{popularity:pop}:{}) };

    if(key==="wide" || key==="place") store[key].push(rec);
    else if(key==="win") store.win = rec;
    else store[key] = rec;
  });

  return store;

  function getCombo($tr){
    // numberSet1_number の並びを読み、間の text (= or -) を見て結合
    const row = $tr.find(".numberSet1_row").first();
    if(!row.length) return tx($tr.find("td").eq(1).text()) || null;

    const nums = row.find(".numberSet1_number").map((i,el)=>tx($(el).text())).get().filter(Boolean);
    const symText = row.text();
    // 記号を判定（= を優先）
    const sym = symText.includes("=") ? "=" : "-";
    if(nums.length===3) return `${nums[0]}${sym}${nums[1]}${sym}${nums[2]}`;
    if(nums.length===2) return `${nums[0]}${sym}${nums[1]}`;
    if(nums.length===1) return nums[0];
    return null;
  }
  function getAmount($tr){ return num(tx($tr.find("td").eq(2).text())); }
  function getPopularity($tr){ const v = num(tx($tr.find("td").eq(3).text())); return Number.isFinite(v)?v:null; }
}

// --- 返還：右側の「返還」テーブルをそのまま（空なら []）
function parseRefunds($){
  const hasRefundHead = $("table:has(thead th:contains('返還'))").length>0;
  if(!hasRefundHead) return [];
  // 公式は返還があると具体の組番が並ぶが、本HTMLは空。代表語のみ返すのはノイズなので空配列で返す。
  return [];
}

async function main(){
  const [,, hd, jcd, rno] = process.argv;
  if(!hd||!jcd||!rno){ console.error("usage: node scripts/fetch-result-official.js YYYYMMDD JCD RNO"); process.exit(1); }

  const outPath = buildOut(hd,jcd,rno);
  if(process.env.SKIP_EXISTING==="true" && fs.existsSync(outPath)){ console.log(`[skip] ${outPath}`); return; }

  const url = `https://www.boatrace.jp/owpc/pc/race/raceresult?rno=${Number(rno)}&jcd=${z2(jcd)}&hd=${hd}`;
  const html = await fetchHtml(url,{retries:3,delay:1200});
  await sleep(Number(process.env.FETCH_DELAY_MS||"900"));
  const $ = load(html);

  const meta0 = parseMeta($);
  const results = parseResults($);
  const { startList, winnerType } = parseStart($);
  const refunds = parseRefunds($);
  const payouts = parsePayouts($);

  // ST補完
  const stMap = new Map(startList.map(o=>[String(o.lane), o.st]));
  for(const r of results){
    if(r.st==null && r.lane && stMap.has(String(r.lane))){
      r.st = Number(stMap.get(String(r.lane))).toFixed(2);
    }
  }
  // 優勝艇に start_type（決まり手優先、なければスタート情報の語）
  const winner = results.find(x=>x.rank==="1");
  if(winner){
    winner.start_type = meta0.decision || winnerType || null;
  }

  const payload = {
    meta: {
      date: hd, jcd: z2(jcd), rno: Number(rno),
      title: meta0.title || null,
      decision: meta0.decision || null,
      weather_sky: meta0.weather_sky || null,
      wind_dir: meta0.wind_dir || null,
      wind_speed_m: meta0.wind_speed_m ?? null,
      wave_height_cm: meta0.wave_height_cm ?? null,
    },
    results,
    start: startList,       // [{lane:number, st:number}]
    refunds,
    payouts,
    source_url: url,
    generated_at: nowISO(),
  };

  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2), "utf-8");
  console.log("[ok]", outPath);
}

main().catch(e=>{ console.error("[error]", e?.message||e); process.exit(1); });
