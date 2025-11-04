#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions向け 高速・安定版：boatrace.jp レース結果スクレイパー
- 環境変数 START_DATE, END_DATE (YYYYMMDD) で日付範囲指定（未指定なら今日）
- 当日の開催場を index から抽出して、その場の 1〜12R だけ巡回
- requests.Session + 適応スリープ + 控えめ並列（MAX_WORKERS）で高速化
- 保存先: data/results/YYYY/MMDD/{jcd}/{rno}R.json
"""

import os, re, time, json, sys
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

BASE_RESULT = "https://www.boatrace.jp/owpc/pc/race/raceresult"
BASE_INDEX  = "https://www.boatrace.jp/owpc/pc/race/index"
HEADERS = {"User-Agent": "Mozilla/5.0 (Actions Bot)", "Accept-Language": "ja,en;q=0.9"}

OUT_BASE = "./data/results"
DEFAULT_SLEEP = float(os.getenv("SLEEP", "1.2"))      # 基本スリープ（適応で0.8〜3.0に変動）
MAX_WORKERS   = int(os.getenv("MAX_WORKERS", "4"))    # 2〜4推奨
MAX_RETRY     = 3
ALLOW_FALLBACK_ALL = os.getenv("ALLOW_FALLBACK_ALL", "false").lower() == "true"  # hosting検出失敗時の全場fallback

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

def clean(s: Optional[str]) -> Optional[str]:
    return " ".join(s.replace("\u3000", " ").split()) if s else None

def text_or_none(el) -> Optional[str]:
    return clean(el.get_text(" ", strip=True)) if el else None

def fetch(session: requests.Session, url: str, sleep_state: Dict[str, float]) -> str:
    """GET with retry & adaptive backoff"""
    backoff = sleep_state.get("sleep", DEFAULT_SLEEP)
    for i in range(MAX_RETRY):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200 and r.text:
                r.encoding = "utf-8"
                # success → 少しずつ短縮（下限0.8）
                sleep_state["sleep"] = max(0.8, backoff * 0.9)
                return r.text
            else:
                print(f"[WARN] GET {url} -> {r.status_code}")
        except Exception as e:
            print(f"[WARN] GET {url} err: {e}")
        # fail → バックオフ（上限3.0）
        backoff = min(3.0, backoff * 1.5)
        sleep_state["sleep"] = backoff
        time.sleep(backoff)
    raise RuntimeError(f"Failed to fetch after retries: {url}")

def get_hosting_jcds(session: requests.Session, date_str: str, sleep_state: Dict[str, float]) -> List[str]:
    """
    /race/index?hd=YYYYMMDD から当日の開催場 jcd を抽出。
    失敗時は ALLOW_FALLBACK_ALL が True のときだけ 01..24 を返す。
    """
    url = f"{BASE_INDEX}?hd={date_str}"
    try:
        html = fetch(session, url, sleep_state)
        soup = BeautifulSoup(html, "lxml")  # HTMLパーサ固定（XML警告回避）
        # aタグの href に含まれる jcd=NN を集める
        jcds = {m.group(1) for a in soup.select("a[href*='jcd=']") if (m := re.search(r"jcd=(\d{2})", a.get("href") or ""))}
        if jcds:
            lst = sorted(jcds)
            print(f"[INFO] {date_str} hosting: {','.join(lst)}")
            return lst
        else:
            print(f"[WARN] {date_str}: hosting not found on index page.")
    except Exception as e:
        print(f"[WARN] {date_str}: hosting detection failed: {e}")
    if ALLOW_FALLBACK_ALL:
        print(f"[WARN] {date_str}: fallback to all 01-24")
        return [f"{i:02d}" for i in range(1, 25)]
    else:
        print(f"[WARN] {date_str}: skip (no hosting detected and fallback disabled)")
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
        if not tds or ("着" in header and "選手" in header):  # ヘッダ行スキップ
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

def scrape_one(session: requests.Session, date: str, jcd: str, rno: int, sleep_state: Dict[str, float]) -> str:
    url = f"{BASE_RESULT}?rno={int(rno)}&jcd={jcd}&hd={date}"
    html = fetch(session, url, sleep_state)
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

    with requests.Session() as sess:
        sleep_state = {"sleep": DEFAULT_SLEEP}
        for day in daterange(sdt, edt):
            dstr = day.strftime("%Y%m%d")
            jcds = get_hosting_jcds(sess, dstr, sleep_state)
            if not jcds:
                continue  # 開催場検出できず＆フォールバック禁止ならスキップ
            print(f"[INFO] {dstr}: {len(jcds)}場 | sleep={sleep_state['sleep']:.2f}s | workers={MAX_WORKERS}")
            tasks = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for jcd in jcds:
                    for rno in range(1, 13):
                        tasks.append(ex.submit(scrape_one, sess, dstr, jcd, rno, sleep_state))
                for fut in as_completed(tasks):
                    try:
                        print("✅", fut.result())
                    except Exception as e:
                        print(f"[WARN] {dstr} failed: {e}", file=sys.stderr)
                    time.sleep(sleep_state.get("sleep", DEFAULT_SLEEP))

if __name__ == "__main__":
    main()
