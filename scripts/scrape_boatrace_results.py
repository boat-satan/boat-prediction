#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vFast3 完全版（場単位並列・開催場自動/手動・堅牢パース）

ENV:
  START_DATE, END_DATE  : YYYYMMDD（未指定は今日）
  JCDS                  : "03,07,12" など指定があればその場のみ
  MAX_WORKERS           : 並列に処理する「場」の数（推奨 4〜6、既定 6）
  SLEEP                 : 同一場内でのリクエスト間隔のベース（既定 1.0 秒）
  ALLOW_FALLBACK_ALL    : hosting検出失敗時に全24場へ落とすなら "true"（既定 false）

出力:
  data/results/YYYY/MMDD/{jcd}/{rno}R.json
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
HEADERS = {"User-Agent": "Mozilla/5.0 (GitHub Actions)", "Accept-Language": "ja,en;q=0.9"}

OUT_BASE = "./data/results"
BASE_SLEEP = float(os.getenv("SLEEP", "1.0"))        # 失敗時に最大2.5まで増える
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))
MAX_RETRY   = 3
ALLOW_FALLBACK_ALL = os.getenv("ALLOW_FALLBACK_ALL", "false").lower() == "true"
JCDS_OVERRIDE = os.getenv("JCDS", "").strip()

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

# ---- utils ----
ZEN2HAN = str.maketrans("０１２３４５６７８９１２３４５６", "0123456789123456")

def clean(s: Optional[str]) -> Optional[str]:
    return " ".join(s.replace("\u3000", " ").split()) if s else None

def norm_num(s: Optional[str]) -> Optional[str]:
    if not s: return s
    return s.translate(ZEN2HAN).strip()

def text_or_none(el) -> Optional[str]:
    return clean(el.get_text(" ", strip=True)) if el else None

def pick_first(patterns, text: str) -> Optional[str]:
    for p in (patterns if isinstance(patterns, (list, tuple)) else [patterns]):
        m = re.search(p, text)
        if m: return m.group(1)
    return None

# ---- network ----
def fetch(url: str, sleep_val: float) -> str:
    """1レース=1セッション（スレッド間共有しない）+軽いバックオフ"""
    backoff = sleep_val
    for _ in range(MAX_RETRY):
        try:
            with requests.Session() as sess:
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

# ---- hosting detection ----
def get_hosting_jcds(date_str: str) -> List[str]:
    # 1) 手動指定があれば最優先
    if JCDS_OVERRIDE:
        lst = sorted({x.zfill(2) for x in re.split(r"[,\s]+", JCDS_OVERRIDE) if x})
        print(f"[INFO] {date_str} using override jcds: {','.join(lst)}")
        return lst

    # 2) indexページから多段抽出
    url = f"{BASE_INDEX}?hd={date_str}"
    try:
        html = fetch(url, BASE_SLEEP)
        soup = BeautifulSoup(html, "lxml")
        jcds = set()
        # パターン1: href に jcd=NN
        for a in soup.select("a[href*='jcd=']"):
            m = re.search(r"jcd=(\d{2})", a.get("href") or "")
            if m: jcds.add(m.group(1))
        # パターン2: data-* 属性等
        if not jcds:
            for tag in soup.find_all(attrs=True):
                for k, v in tag.attrs.items():
                    if isinstance(v, str) and re.fullmatch(r"\d{2}", v) and ("jcd" in k.lower() or "place" in k.lower()):
                        jcds.add(v)
        # パターン3: script 内の JSON 片
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

# ---- parsing ----
def parse_weather_block(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    t = soup.get_text(" ", strip=True)
    sky  = pick_first([r"天候\s*([^\s/|・]+)", r"天候：?\s*([^\s/|・]+)"], t)
    ws   = pick_first([r"風[^0-9]*?(\d+(?:\.\d+)?)\s*m", r"風速：?\s*(\d+(?:\.\d+)?)\s*m/?s?"], t)
    wh   = pick_first([r"波[^0-9]*?(\d+)\s*cm", r"波高：?\s*(\d+)\s*cm"], t)
    WIND_DIRS = ["向い風","向かい風","追い風","右横風","左横風","北","北東","東","南東","南","南西","西","北西"]
    wind_dir = next((w for w in WIND_DIRS if w in t), None)
    if not wind_dir:
        alt = " ".join(img.get("alt","") for img in soup.find_all("img"))
        wind_dir = next((w for w in WIND_DIRS if w in alt), None)
    return {
        "weather_sky": sky,
        "wind_dir": wind_dir,
        "wind_speed_m": float(ws) if ws else None,
        "wave_height_cm": int(wh) if wh else None
    }

def parse_meta(soup: BeautifulSoup, date: str, jcd: str, rno: int) -> RaceMeta:
    m = RaceMeta(date=date, jcd=jcd, rno=rno)
    if (h := soup.select_one("h2")):
        m.title = text_or_none(h)
    # 決まり手はワードだけ保持（".11 抜き" → "抜き"）
    body = soup.get_text(" ", strip=True)
    if body:
        dec = pick_first([r"(逃げ|差し|まくり差し|まくり|抜き|恵まれ|叩き)"], body)
        m.decision = dec
    w = parse_weather_block(soup)
    m.weather_sky, m.wind_dir, m.wind_speed_m, m.wave_height_cm = (
        w["weather_sky"], w["wind_dir"], w["wind_speed_m"], w["wave_height_cm"]
    )
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
    # ヘッダ→列インデックス
    head_row = tbl.find("tr")
    ths = head_row.find_all(["th","td"]) if head_row else []
    head_texts = [ (text_or_none(th) or "") for th in ths ]
    COLS = {
        "rank":  ["着","着順"],
        "lane":  ["艇","艇番"],
        "name":  ["選手","選手名"],
        "course":["コース","進入","進入コース"],
        "st":    ["ST","スタート","スタートタイミング"],
        "time":  ["タイム","着順タイム","決まり手タイム"],
        "note":  ["備考","返還","決","決まり手"]
    }
    col_idx = {}
    for i, ht in enumerate(head_texts):
        for key, aliases in COLS.items():
            if any(alias in ht for alias in aliases):
                col_idx.setdefault(key, i)

    rows: List[ResultRow] = []
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if not tds: 
            continue
        cells = [ text_or_none(td) or "" for td in tds ]
        joined = "".join(cells)
        # 明らかなヘッダ/ダミー行は除外
        if "選手" in joined and "着" in joined: continue
        if "ボートレーサー" in joined and len(cells) <= 2: continue

        def pick(col, fb=None):
            if col in col_idx and col_idx[col] < len(cells):
                return cells[col_idx[col]]
            return fb

        rank = pick("rank") or None
        lane = norm_num(pick("lane"))
        name = pick("name") or ""

        rid, name2 = extract_racer_id(name)
        racer_id = rid
        racer_name = name2 or name or None

        st = pick("st") or None
        time_ = pick("time") or None
        course = norm_num(pick("course")) or None
        note = pick("note") or None

        start_type = None
        if st:
            st = st.strip()
            if st and st[0] in ("F","L"):
                start_type, st = st[0], st[1:]
            st = st.strip() or None

        if not (racer_name or lane or rank):
            continue

        rows.append(ResultRow(
            rank=rank, lane=lane,
            racer_id=racer_id, racer_name=racer_name,
            st=st, course=course, time=time_,
            start_type=start_type, note=note
        ))

    # たまに先頭に来るダミー「ボートレーサー」行を除去
    rows = [r for r in rows if (r.racer_name and r.racer_name != "ボートレーサー")]
    return rows

# ---- scraping ----
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
    """同一場の1〜12Rを同スレッドで直列処理（場は並列）"""
    saved = []
    sleep_val = BASE_SLEEP
    for rno in range(1, 13):
        try:
            saved.append(scrape_one_race(date, jcd, rno, sleep_val))
        except Exception as e:
            print(f"[WARN] {date} {jcd}-{rno}R: {e}")
        time.sleep(sleep_val)  # 同一場内のレート制御
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
        workers = min(MAX_WORKERS, len(jcds))
        print(f"[INFO] {dstr}: target venues={len(jcds)} | workers={workers} | base_sleep={BASE_SLEEP}")
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for jcd in jcds:
                print(f"[INFO] start jcd={jcd}")
                futures.append(ex.submit(scrape_one_jcd, dstr, jcd))
            for fut in as_completed(futures):
                try:
                    for p in fut.result():
                        print(f"✅ {p}")
                except Exception as e:
                    print(f"[WARN] {dstr} jcd task failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
