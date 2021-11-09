from pathlib import Path
from itertools import groupby

lang_map = {}
BASE_DIR = Path(__file__).parent.absolute().resolve()

def get_lang(mp3_name, lang_path=None):    
    global lang_map

    if not lang_map:  
        if not lang_path:
            lang_path = BASE_DIR / "../../data/lang_map.csv"          
        with open(lang_path, "r") as fin:
            for ln in fin.readlines():
                fname, lang = ln.strip().split(",")
                lang_map[fname] = lang
    return lang_map.get(Path(mp3_name).name, "none")

def get_segments(mp3_name, seg_dir=None):
    if not seg_dir:
        seg_dir = BASE_DIR / "../../data/segments"
        seg_path = seg_dir / Path(Path(mp3_name).stem+".seg.csv")
    
    if not seg_path.exists():
        return []

    with open(seg_path, "r") as fin:
        segs = [x.strip().split(",") for x in fin.readlines()]   
        grp_iter = groupby(segs, key=lambda x: x[2])
        grouped_segs = []
        for grp_key, grp_value in grp_iter:
            group = list(grp_value)
            grp_start = group[0][0]
            grp_end = group[-1][1]
            grouped_segs.append((grp_start, grp_end, grp_key))
    
    return grouped_segs

def get_segment_type(mp3_name, start, end, seg_dir=None):

    best = 0
    candid = ("none", best)
    grouped_segs = get_segments(mp3_name, seg_dir)

    for ss, ee, cat in grouped_segs:
        ss = int(ss)
        ee = int(ee)
        if ee < start: continue  # not yet there
        if ss > end: break       # past the mark
                        
        int_start = max(ss, start)
        int_end = min(ee, end)
        duration = int_end-int_start

        if best < duration:
            candid = (cat, duration)
            best = duration
    
    return candid
            
    

    
