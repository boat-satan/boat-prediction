#!/usr/bin/env node
/**
 * BOATRACE 結果スクレイパ (null最小化)
 * usage: node scripts/fetch-result-official.js YYYYMMDD JCD RNO
 * out: data/results/YYYY/MMDD/{jcd}/{rno}R.json
 * schema: ユーザー提示JSONに準拠
 */

import fs from "node:fs";
import path from "node:path";
import { load } from "cheerio";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const z2 = (n) => String(n).padStart(2, "0");
const nowISO = () => new Date().toISOString();

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function toHalf(s=""){ return String(s).replace(/[０-９Ａ-Ｚａ-ｚ．－−，：’”]/g, ch=>{
  const map={ "．":".","－":"-","−":"-","，":",","：":":","’":"'","”":"\"" }; if(map[ch]) return map[ch];
  return String.fromCharCode(ch.charCodeAt(0)-0xFEE0);
});}
function T($el){ return toHalf($el.text().replace(/\s+/g," ").trim()); }
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

// --- Meta (天候/風/波/決まり手/タイトル)
function parseMeta($){
  const meta = {};
  meta.title = T($(".hdg2, .heading2, .result_hd, .raceTitle").first());
  const ctx = $(".weather1, .weather, .result_table, .result_info, .raceInfo, .result_info2").first().text();
  const s = toHalf(ctx);

  const mDec = s.match(/決まり手\s*[:：]?\s*([^\n\r\t 　]+)/);
  const mSky = s.match(/天候\s*[:：]?\s*([^\n\r\t 　]+)/);
  const mWind = s.match(/風\s*[:：]?\s*([東西南北南北東西南西北東北西]{1,3}|[\u4E00-\u9FFF]+)\s*(\d+(?:\.\d+)?)?m/);
  const mWave = s.match(/波\s*[:：]?\s*(\d+(?:\.\d+)?)m/);

  meta.decision = mDec?.[1] ?? null;
  meta.weather_sky = mSky?.[1] ?? null;
  meta.wind_dir = mWind?.[1] ?? null;
  meta.wind_speed_m = mWind?.[2] ? Number(mWind[2]) : null;
  meta.wave_height_cm = mWave ? Math.round(Number(mWave[1]) * 100) : null;

  return meta;
}

// ヘッダ名→列index を作る
function headerIndex($table){
  const map = {};
  $table.find("tr").first().find("th,td").each((i,th)=>{
    const k=T($(th));
    map[k]=i;
  });
  return map;
}

// 結果テーブル（着/艇/選手/登番/ST/進入/タイム/備考）
function parseResultsTable($){
  let target=null, headStr="";
  $("table").each((_,el)=>{
    const $t=$(el);
    const heads=$t.find("th").map((i,th)=>T($(th))).get().join("|");
    if(/着|着順/.test(heads) && /艇|選手|登番/.test(heads)){ target=$t; headStr=heads; }
  });
  if(!target) return { rows:[], head:{} };

  const head = headerIndex(target);
  const keys = Object.fromEntries(Object.entries(head).map(([k,i])=>[k.replace(/\s/g,""),i]));

  const rows=[];
  target.find("tr").slice(1).each((_,tr)=>{
    const $tr=$(tr); const tds=$tr.find("td"); if(!tds.length) return;
    const get=(nameRegex)=>{ // th名部分一致
      const hit = Object.entries(keys).find(([k])=>nameRegex.test(k));
      if(!hit) return null;
      const idx = hit[1];
      return T($(tds.get(idx) ?? []));
    };

    const rank = get(/着|着順/);
    const lane = get(/艇/);
    // 登番は数字4桁
    let racer_id = get(/登番|登録|番号|No/);
    if(racer_id && !/\d{4}/.test(racer_id)) racer_id = (racer_id.match(/\d{4}/)||[])[0] ?? null;

    let racer_name = get(/選手|氏名|名前/);
    if(racer_name) racer_name = racer_name.replace(/\d{4}.*/,"").trim();

    let st = get(/ST|スタート/);
    // ST 見栄え統一（F.02/L.01/0.13）
    if(st){
      const m=st.match(/^(F|L)?\.?(\d?\.\d{2})$/);
      if(m) st = (m[1]?`${m[1]}.`:"")+m[2];
      else if(/^\d\.\d{2}$/.test(st)) st=st;
      else st=null;
    }

    let course = get(/進入|コース/);
    // 行ごとの進入（1～6）や、全体の「123/456」などが入ることがある
    if(course && !/^([1-6]{1,6}(\/[1-6]{1,6})?|[1-6])$/.test(course)) course=null;

    let time = get(/ﾀｲﾑ|タイム|決着/);
    // 1'51"0 / 1:51.0 / 1.51.0 などを 1'51"0 に寄せる軽整形
    if(time){
      const t = time.replace(/[：:]/g,":").replace(/[″”"]/g,'"').replace(/[′’']/g,"'");
      const m = t.match(/^(\d)[:.](\d{2})[."”](\d)$/) || t.match(/^(\d)['’](\d{2})["”](\d)$/);
      if(m) time = `${m[1]}'${m[2]}"${m[3]}`;
    }

    let note = get(/備考|状況|事故/);
    if(note && !note.replace(/[-–—]/g,"").trim()) note=null;

    rows.push({
      rank: rank || null,
      lane: lane || null,
      racer_id: racer_id || null,
      racer_name: racer_name || null,
      st: st || null,
      course: course || null,
      time: time || null,
      start_type: null, // 公式に明示があれば後で埋める
      note: note || null,
    });
  });

  return { rows, head: keys, headStr };
}

// ST一覧（艇→ST）
function parseStartList($){
  const out=[];
  $("table").each((_,el)=>{
    const $t=$(el);
    const heads=$t.find("th").map((i,th)=>T($(th))).get().join("|");
    if(/ST/i.test(heads) && /艇|進入|枠/.test(heads)){
      const head = headerIndex($t);
      const laneIdx = Object.entries(head).find(([k])=>/艇|枠/.test(k))?.[1] ?? 0;
      let stIdx = Object.entries(head).find(([k])=>/ST/i.test(k))?.[1];
      $t.find("tr").slice(1).each((_,tr)=>{
        const tds=$(tr).find("td"); if(!tds.length) return;
        const lane=T($(tds.get(laneIdx)||[]));
        const stRaw = stIdx!=null ? T($(tds.get(stIdx)||[])) : "";
        let st = null;
        const m=stRaw.match(/^(F|L)?\.?(\d?\.\d{2})$/);
        if(m) st = Number((m[1]?`-${m[2]}`:m[2])); // 数値に（F/Lは数値化困難なので負数で区別or後で文字に戻す）
        else if(/^\d\.\d{2}$/.test(stRaw)) st = Number(stRaw);
        if(/^[1-6]$/.test(lane) && st!=null) out.push({ lane:Number(lane), st:Number(st.toFixed(2)) });
      });
    }
  });
  // 重複 lane は最初を優先
  const uniq = new Map();
  for(const r of out){ if(!uniq.has(r.lane)) uniq.set(r.lane,r); }
  return [...uniq.values()];
}

// 配当（勝式ごと）
function parsePayouts($){
  // 勝式名の正規化
  const normKind = (s)=>{
    s = s.replace(/\s+/g,"");
    if(/3連単|三連単/.test(s)) return "trifecta";
    if(/3連複|三連複/.test(s)) return "trio";
    if(/2連単|二連単/.test(s)) return "exacta";
    if(/2連複|二連複/.test(s)) return "quinella";
    if(/拡連複|ワイド|拡大二連複/.test(s)) return "wide";
    if(/単勝/.test(s)) return "win";
    if(/複勝/.test(s)) return "place";
    return null;
  };

  const store = {
    trifecta:null, trio:null, exacta:null, quinella:null,
    wide:[], win:null, place:[]
  };

  $("table").each((_,el)=>{
    const $t=$(el);
    const heads=$t.find("th").map((i,th)=>T($(th))).get().join("|");
    if(!/払戻|配当|勝式/.test(heads)) return;

    $t.find("tr").slice(1).each((_,tr)=>{
      const tds=$(tr).find("td");
      if(tds.length<2) return;

      const kindRaw=T($(tds.get(0)));
      const kind=normKind(kindRaw);
      if(!kind) return;

      const combo=T($(tds.get(1)));
      const amount=num(T($(tds.get(2))));
      const pop = tds.get(3)? num(T($(tds.get(3)))) : null;
      if(!combo || amount==null) return;

      const rec = { combo, amount, popularity: pop ?? undefined };
      if(kind==="wide" || kind==="place"){
        store[kind].push(rec);
      }else if(kind==="win"){
        store.win = rec;
      }else{
        store[kind]=rec;
      }
    });
  });

  // popularity undefined は出力しない（JSON.stringifyで残るので整理）
  const clean = (o)=>{
    if(Array.isArray(o)) return o.map(clean);
    if(o && typeof o === "object"){
      const r={};
      for(const [k,v] of Object.entries(o)){
        if(v===undefined) continue;
        r[k]=clean(v);
      }
      return r;
    }
    return o;
  };

  return clean(store);
}

// 返還など
function parseRefunds($){
  const txt = toHalf($("body").text());
  const out = [];
  for(const line of txt.split(/\n+/)){
    const s=line.trim();
    if(!s) continue;
    if(/返還|不成立|没収/.test(s)) out.push(s);
  }
  return out;
}

async function main(){
  const [,, hd, jcd, rno] = process.argv;
  if(!hd||!jcd||!rno){ console.error("usage: node scripts/fetch-result-official.js YYYYMMDD JCD RNO"); process.exit(1); }

  const outPath = buildOut(hd,jcd,rno);
  if(process.env.SKIP_EXISTING==="true" && fs.existsSync(outPath)){ console.log(`[skip] ${outPath}`); return; }

  const url = `https://www.boatrace.jp/owpc/pc/race/raceresult?rno=${Number(rno)}&jcd=${z2(jcd)}&hd=${hd}`;
  const html = await fetchHtml(url,{retries:3,delay:1200});
  await sleep(Number(process.env.FETCH_DELAY_MS||"1200"));
  const $ = load(html);

  // メタ
  const meta0 = parseMeta($);
  const meta = {
    date: hd,
    jcd: z2(jcd),
    rno: Number(rno),
    title: meta0.title || null,
    decision: meta0.decision || null,
    weather_sky: meta0.weather_sky || null,
    wind_dir: meta0.wind_dir || null,
    wind_speed_m: meta0.wind_speed_m ?? null,
    wave_height_cm: meta0.wave_height_cm ?? null,
  };

  // 結果
  const { rows, headStr } = parseResultsTable($);

  // ST一覧から補完
  const startList = parseStartList($); // [{lane:Number, st:Number}]
  const stMap = new Map(startList.map(o=>[String(o.lane), o.st])); // lane->number

  for(const r of rows){
    // ST補完（結果表に無ければ startList から）
    if(r.st==null && r.lane && stMap.has(String(r.lane))){
      const stNum = stMap.get(String(r.lane));
      r.st = Number.isFinite(stNum) ? Number(stNum).toFixed(2) : null; // 文字列 "0.13"
    }
  }

  // 進入（全体進入が1セルにある場合は、各艇のcourse未設定なら自艇進入番号を推定）
  let wholeEntry = null;
  const bodyTxt = toHalf($("body").text());
  const mm = bodyTxt.match(/\b([1-6]{1,6})(?:\/([1-6]{1,6}))?\b/);
  if(mm){
    wholeEntry = mm[2] ? `${mm[1]}/${mm[2]}` : mm[1];
  }
  if(wholeEntry){
    const seq = wholeEntry.split("/").join("").split(""); // ["1","2","3","4","5","6"] など
    const laneToCourse = new Map(seq.map((ln,i)=>[ln, String(i+1)]));
    for(const r of rows){
      if(!r.course && r.lane && laneToCourse.has(String(r.lane))){
        r.course = laneToCourse.get(String(r.lane)); // "1".."6"
      }
    }
  }

  // start_type（ここは公式の有無に依存。決まり手が「逃げ/まくり/差し/まくり差し/抜き/恵まれ」であれば winner にのみ付与）
  const winner = rows.find(x=>x.rank==="1");
  if(winner && meta.decision){
    winner.start_type = meta.decision; // まずは決まり手を start_type に反映（用途に応じてリネーム可）
  }

  const payouts = parsePayouts($);
  const refunds = parseRefunds($);

  const payload = {
    meta,
    results: rows,
    start: startList, // 数値 (lane:number, st:number)
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
