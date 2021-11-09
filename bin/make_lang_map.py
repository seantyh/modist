from datetime import datetime
from pathlib import Path
from glob import glob

BASE_DIR = Path(__file__).parent.absolute()
with (BASE_DIR/ "../data/50_week_5008_lang.txt").open("r") as fin:
    lang_table = fin.readlines()

lang_map = filter(lambda x: x.startswith("5008"), lang_table)
lang_map = map(lambda x: x.strip().split(","), lang_map)
lang_map = map(lambda x: (*x[0].split("-")[2:4], x[1]), lang_map)
lang_map = map(lambda x: ((x[0], x[1]), x[2]), lang_map)
lang_map = dict(lang_map)

mp3_files = glob(str(BASE_DIR/"../data/mp3/*.mp3"))
fout = (BASE_DIR / "../data/lang_map.csv").open("w")
for fname in sorted(mp3_files):
    fname = Path(fname).name
    dtime = datetime.strptime(fname, "c5008-%y%m%d%H%M.mp3")
    dt_key = (dtime.strftime("%a"), dtime.strftime("%H%M"))
    fout.write(f"{fname},{lang_map.get(dt_key)}\n")
fout.close()
