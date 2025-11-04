#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vFast3: 場単位並列版（GH Actions最適）
- START_DATE, END_DATE (YYYYMMDD)
- JCDS="03,07,12" を指定するとその場のみ
- hosting検出失敗時は ALLOW_FALLBACK_ALL=true で 01..24 へ
"""

import os, re, time, json, sys, warnings
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BASE_RESULT = "https://www.boatrace.jp/owpc/pc/race/raceresult"
BASE_INDEX  = "https://www.boatrace.jp/owpc/pc/race/index"
HEADERS = {"User-Agent": "Mozilla/5.0 (Actions Bot)", "Accept-Language": "ja,en;q=0.9"}

OUT_BASE = "./data/results"
BASE_SLEEP = float(os.getenv("SLEEP", "1.0"))      # 1.0を基準、失敗時に最大2.5へ
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))   # 場を並列（まずは4〜6）
MAX_RETRY   = 3
ALLOW_FALLBACK_ALL = os.getenv("ALLOW_FALLBACK_ALL", "false").lower() == "true"
JCDS_OVERRIDE = os.getenv("JCDS", "").strip()      # "03,07,12" など

@dataclass
class RaceMeta:
    date: str
    jcd: str
    rno: int
    title: Optional[str] = None
    decision: Optional[str] = None
    weather_sky: Optional[str] = None
    wind_dir: Optional[str] = None
    wind_speed_m: Optional[float] = None
    wave_height_cm: Optional[int] = None

@dataclass
class ResultRow:
    rank: Optional[str]
    lane: Optional[str]
    racer_id: Optional[str]
    racer_name: Optional[str]
    st: Optional[str]
    course: Optional[str]
    time: Optional[str]
    start_type: Optional[str]
    note: Optional[str]

def clean(s): 
    return " ".join(s.replace("\u3000", " ").split()) if s else None

def text_or_none(el):
    return clean(el.get_text(" ", strip=True)) if el else None

def fetch(url: str, sleep_val: float) -> str:
    backoff = sleep_val
    for i in range(MAX_RETRY):
        try:
            with requests.Session() as sess:  # ← タスク毎セッション（共有しない）
                r = sess.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200 and r.text:
                r.encoding = "utf-8"
                return r.text
            else:
                print(f"[WARN] GET {url} -> {r.status_code}")
        except Exception as e:
            print(f"[WARN] GET {url} err: {e}")
        backoff = min(2.5, backoff * 1.5)
        time.sleep(backoff)
    raise RuntimeError(f"Failed after {MAX_RETRY} retries: {url}")

def get_hosting_jcds(date_str: str) -> List[str]:
    # 1) 手動指定あればそれを最優先
    if JCDS_OVERRIDE:
        lst = sorted({x.zfill(2) for x in re.split(r"[,\s]+", JCDS_OVERRIDE) if x})
        print(f"[INFO] {date_str} using override jcds: {','.join(lst)}")
        return lst

    # 2) indexページから抽出（多段）
    url = f"{BASE_INDEX}?hd={date_str}"
    try:
        html = fetch(url, BASE_SLEEP)
        soup = BeautifulSoup(html, "lxml")
        jcds = set()
        # パターン1: href=jcd=NN
        for a in soup.select("a[href*='jcd=']"):
            m = re.search(r"jcd=(\d{2})", a.get("href") or "")
            if m: jcds.add(m.group(1))
        # パターン2: data属性/テキストに2桁コード
        if not jcds:
            for tag in soup.find_all(attrs=True):
                for k, v in tag.attrs.items():
                    if isinstance(v, str) and re.fullmatch(r"\d{2}", v) and ("jcd" in k.lower() or "place" in k.lower()):
                        jcds.add(v)
        # パターン3: script内のJSON片
        if not jcds:
            for sc in soup.select("script"):
                t = sc.get_text() or ""
                for m in re.finditer(r'jcd["\']?\s*[:=]\s*["\']?(\d{2})["\']?', t):
                    jcds.add(m.group(1))
        if jcds:
            lst = sorted(jcds)
            print(f"[INFO] {date_str} hosting: {','.join(lst)}")
            return lst
        print(f"[WARN] {date_str}: hosting not found on index page.")
    except Exception as e:
        print(f"[WARN] {date_str}: hosting detection failed: {e}")

    # 3) フォールバック
    if ALLOW_FALLBACK_ALL:
        print(f"[WARN] {date_str}: fallback to ALL 01-24")
        return [f"{i:02d}" for i in range(1, 25)]
    else:
        print(f"[WARN] {date_str}: skip (no hosting & fallback disabled)")
        return []

def parse_weather_block(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    WIND_DIRS = ["向い風","向かい風","追い風","右横風","左横風","北","北東","東","南東","南","南西","西","北西"]
    t = soup.get_text(" ", strip=True)
    m_ws = re.search(r"風[^0-9]*?(\d+(?:\.\d+)?)\s*m", t)
    m_wh = re.search(r"波[^0-9]*?(\d+)\s*cm", t)
    m_sky = re.search(r"天候\s*([^\s/|・]+)", t)
    wd = next((k for k in WIND_DIRS if k in t), None)
    return {
        "weather_sky": m_sky.group(1) if m_sky else None,
        "wind_dir": wd,
        "wind_speed_m": float(m_ws.group(1)) if m_ws else None,
        "wave_height_cm": int(m_wh.group(1)) if m_wh else None
    }

def parse_meta(soup: BeautifulSoup, date: str, jcd: str, rno: int) -> RaceMeta:
    m = RaceMeta(date=date, jcd=jcd, rno=rno)
    if (h := soup.select_one("h2")): m.title = text_or_none(h)
    if (d := soup.find(string=re.compile("逃げ|差し|まくり|抜き|恵まれ"))): m.decision = clean(d)
    w = parse_weather_block(soup)
    m.weather_sky, m.wind_dir, m.wind_speed_m, m.wave_height_cm = w["weather_sky"], w["wind_dir"], w["wind_speed_m"], w["wave_height_cm"]
    return m

def guess_result_table(soup: BeautifulSoup):
    best, score = None, -1
    for t in soup.select("table"):
        header = " ".join(h.get_text(" ", strip=True) for h in t.select("thead,th"))
        sc = sum(k in header for k in ["着","艇","選手","ST","コース","タイム"])
        if sc > score: best, score = t, sc
    return best

def extract_racer_id(txt: str):
    t = clean(txt or "")
    if not t: return None, None
    m = re.match(r"^\s*(\d{4})\s*(.*)$", t)
    if m: return m.group(1), m.group(2).strip("（）() ")
    return None, t

def parse_results(soup: BeautifulSoup) -> List[ResultRow]:
    tbl = guess_result_table(soup)
    if not tbl: return []
    rows: List[ResultRow] = []
    for tr in tbl.select("tr"):
        tds = [text_or_none(td) for td in tr.find_all(["td","th"])]
        header = "".join([t or "" for t in tds])
        if not tds or ("着" in header and "選手" in header):
            continue
        rank=lane=rid=name=st=course=time_=sttype=note=None
        for x in tds:
            if not x: continue
            if x in list("123456欠妨失不"): rank=x
            if x.isdigit() and 1<=int(x)<=6 and lane is None: lane=x
            if (x.startswith("F") or x.startswith("L")) and "." in x: sttype=x[0]; st=x[1:]
            elif x.startswith("0.") and st is None: st=x
            elif ":" in x: time_=x
            elif any(k in x for k in ["返還","妨害","失格","転覆","欠場","沈没"]): note=x
        cand = sorted(tds, key=lambda s: -(len(s or "")))
        if cand:
            rid2, name2 = extract_racer_id((cand[0] or ""))
            rid = rid2 or rid; name = name2 or name
        rows.append(ResultRow(rank,lane,rid,name,st,course,time_,sttype,note))
    return [r for r in rows if r.racer_name or r.lane or r.rank]

def scrape_one_race(date: str, jcd: str, rno: int, sleep_val: float) -> str:
    url = f"{BASE_RESULT}?rno={int(rno)}&jcd={jcd}&hd={date}"
    html = fetch(url, sleep_val)
    soup = BeautifulSoup(html, "lxml")
    meta = parse_meta(soup, date, jcd, rno)
    data = {"meta": asdict(meta), "results": [asdict(r) for r in parse_results(soup)], "source_url": url}
    year, md = date[:4], date[4:]
    base = os.path.join(OUT_BASE, year, md, jcd)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{rno}R.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def scrape_one_jcd(date: str, jcd: str) -> List[str]:
    """同一場の1〜12Rを**同じスレッド**で処理（場単位並列）"""
    saved = []
    sleep_val = BASE_SLEEP
    for rno in range(1, 13):
        try:
            p = scrape_one_race(date, jcd, rno, sleep_val)
            saved.append(p)
        except Exception as e:
            print(f"[WARN] {date} {jcd}-{rno}R: {e}")
        # 軽いレート制御（場内でのみ）
        time.sleep(sleep_val)
    return saved

def daterange(s, e):
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)

def main():
    s = os.getenv("START_DATE"); e = os.getenv("END_DATE")
    if not s or not e:
        today = datetime.now().strftime("%Y%m%d"); s = e = today
    sdt = datetime.strptime(s, "%Y%m%d"); edt = datetime.strptime(e, "%Y%m%d")

    for day in daterange(sdt, edt):
        dstr = day.strftime("%Y%m%d")
        jcds = get_hosting_jcds(dstr)
        if not jcds:
            continue
        print(f"[INFO] {dstr}: target venues={len(jcds)} | workers={min(MAX_WORKERS,len(jcds))}")
        # 場を並列に
        futures = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jcds))) as ex:
            for jcd in jcds:
                print(f"[INFO] start jcd={jcd}")
                futures.append(ex.submit(scrape_one_jcd, dstr, jcd))
            for fut in as_completed(futures):
                try:
                    paths = fut.result()
                    for p in paths:
                        print(f"✅ {p}")
                except Exception as e:
                    print(f"[WARN] {dstr} jcd task failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
