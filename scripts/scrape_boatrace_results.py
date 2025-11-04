# scrape_boatrace_results.py
# usage:
#   python scrape_boatrace_results.py --date 20250809 --jcd 04 --rno 1 2 3 4 5 6 7 8 9 10 11 12 --base-dir ./data/results
#   # => ./data/results/2025/0809/04/1R.json などに自動保存

import argparse, os, json, time, sys
import requests, re
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

BASE = "https://www.boatrace.jp/owpc/pc/race/raceresult"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "ja,en;q=0.9"}

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

def fetch_html(url: str) -> str:
    for i in range(3):
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            r.encoding = "utf-8"
            return r.text
        time.sleep(1.5)
    raise RuntimeError(f"fetch failed: {url}")

def clean(s): return " ".join(s.replace("\u3000", " ").split()) if s else None
def text_or_none(el): return clean(el.get_text(" ", strip=True)) if el else None

WIND_DIR_WORDS = ["向い風","追い風","右横風","左横風","北","北東","東","南東","南","南西","西","北西"]
def parse_weather_block(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    cand_blocks = soup.select(".weather1,.weather2,.weatherWrap,.result_info,.winfo,.weatherInfo,.raceInfo") or [soup.body]
    sky = wd = None; ws = wh = None
    re_wind_speed = re.compile(r"風[^0-9]*?(\d+(?:\.\d+)?)\s*m")
    re_wave_cm = re.compile(r"波[^0-9]*?(\d+)\s*cm")
    re_weather_sky = re.compile(r"天候\s*([^\s/|・]+)")
    for b in cand_blocks:
        t = text_or_none(b) or ""
        if not t: continue
        if not sky and (m:=re_weather_sky.search(t)): sky = m.group(1)
        if ws is None and (m:=re_wind_speed.search(t)): ws = float(m.group(1))
        if wh is None and (m:=re_wave_cm.search(t)): wh = int(m.group(1))
        if not wd:
            for k in WIND_DIR_WORDS:
                if k in t: wd = k; break
        if not wd:
            for a in [i.get("alt") for i in b.select("img[alt]") if i.get("alt")]:
                for k in WIND_DIR_WORDS:
                    if k in a: wd = k; break
        if sky and wd and ws is not None and wh is not None: break
    return {"weather_sky": sky, "wind_dir": wd, "wind_speed_m": ws, "wave_height_cm": wh}

def parse_meta(soup, date, jcd, rno):
    meta = RaceMeta(date=date, jcd=jcd, rno=rno)
    if (h:=soup.select_one("h2")): meta.title = text_or_none(h)
    if (de:=soup.find(string=re.compile("逃げ|差し|まくり|抜き|恵まれ"))):
        meta.decision = clean(de)
    w = parse_weather_block(soup)
    meta.weather_sky,meta.wind_dir,meta.wind_speed_m,meta.wave_height_cm = w.values()
    return meta

def guess_result_table(soup):
    tables = soup.select("table")
    keys = ["着","艇","選手","ST","コース","タイム"]
    best,bestscore=None,-1
    for t in tables:
        header=" ".join(h.get_text(" ",strip=True) for h in t.select("thead,th"))
        s=sum(k in header for k in keys)
        if s>bestscore: best,bestscore=t,s
    return best

def extract_racer_id(txt):
    t=clean(txt or ""); 
    if not t: return None,None
    if (m:=re.match(r"^\s*(\d{4})\s*(.*)$",t)):
        return m.group(1), m.group(2).strip("（）() ")
    return None,t

def parse_results(soup)->List[ResultRow]:
    tbl=guess_result_table(soup)
    if not tbl: return []
    rows=[]
    for tr in tbl.select("tr"):
        tds=[text_or_none(td) for td in tr.find_all(["td","th"])]
        if not tds or "着" in "".join(tds): continue
        rank=lane=rid=name=st=course=time_=sttype=note=None
        for x in tds:
            if not x: continue
            if x in list("123456欠妨失不"): rank=x
            if x.isdigit() and 1<=int(x)<=6: lane=x
            if (x.startswith("F") or x.startswith("L")) and "." in x:
                sttype=x[0]; st=x[1:]
            elif x.startswith("0."): st=x
            elif ":" in x: time_=x
            elif any(k in x for k in["返還","妨害","失格","転覆","欠場"]): note=x
        rid,name=extract_racer_id(tds[2] if len(tds)>2 else None)
        rows.append(ResultRow(rank,lane,rid,name,st,course,time_,sttype,note))
    return rows

def scrape_one(date,jcd,rno):
    url=f"{BASE}?rno={rno}&jcd={jcd}&hd={date}"
    soup=BeautifulSoup(fetch_html(url),"lxml")
    meta=parse_meta(soup,date,jcd,rno)
    return {
        "meta":asdict(meta),
        "results":[asdict(r) for r in parse_results(soup)],
        "source_url":url
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--date",required=True)
    ap.add_argument("--jcd",required=True)
    ap.add_argument("--rno",required=True,nargs="+",type=int)
    ap.add_argument("--base-dir",default="./data/results")
    args=ap.parse_args()

    year=args.date[:4]; monthday=args.date[4:]
    base=os.path.join(args.base_dir,year,monthday,args.jcd)
    os.makedirs(base,exist_ok=True)

    for rno in args.rno:
        try:
            d=scrape_one(args.date,args.jcd,rno)
            outpath=os.path.join(base,f"{rno}R.json")
            with open(outpath,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2)
            print(f"✅ saved: {outpath}")
            time.sleep(0.8)
        except Exception as e:
            print(f"[WARN] {args.jcd}-{rno}R failed: {e}",file=sys.stderr)

if __name__=="__main__":
    main()
