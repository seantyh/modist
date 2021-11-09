import random
from pathlib import Path

def split_mp3():
    BASE_DIR = Path(__file__).parent.absolute().resolve()
    random.seed(12345)
    lang_path = BASE_DIR / "../../data/lang_map.csv"
    lang_data = {}
    with open(lang_path, "r") as fin:
        for ln in sorted(fin.readlines()):
            fname, lang = ln.strip().split(",")
            lang_data.setdefault(lang, []).append(fname)
    
    for lang, ldata in lang_data.items():        
        random.shuffle(ldata)
    
    train_files = []
    test_files = []

    for lang, ldata in lang_data.items():
        train_files.extend((x, lang) for x in ldata[1:])
        test_files.append((ldata[0], lang))
        
    return train_files, test_files
