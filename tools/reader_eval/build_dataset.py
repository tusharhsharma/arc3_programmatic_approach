"""THE READER TRIAL — the dataset.  For every public game's first frame
(and the first frame of each second level the core has recorded), the
scene as the reader sees it (agent/reader.py: things table + grid), the
exact prompt an LLMReader would send, and the gold reading a person
makes (study/reader_eval/gold.json, written from the human readings).

    .venv/bin/python tools/reader_eval/build_dataset.py      -> study/reader_eval/dataset.json

The dataset is what the Kaggle notebook (study/reader_eval/reader_trial.ipynb)
runs a model over; nothing in it needs a GPU to build.
"""
import os, sys, json, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
logging.disable(logging.INFO)
import numpy as np
import arc_agi
from agent.perception import background_color
from agent.core.scene import SceneTracker
from agent.reader import LLMReader, scene_text

PUBLIC = "tn36 lf52 cn04 bp35 wa30 lp85 r11l tu93 sp80 m0r0 vc33 ar25 ka59 sc25 sk48 dc22 cd82 ft09 g50t ls20 re86 s5i5 sb26 su15 tr87".split()
OUT = ROOT / "study" / "reader_eval"
OUT.mkdir(parents=True, exist_ok=True)
gold = json.load(open(OUT / "gold.json")) if (OUT / "gold.json").exists() else {}
reader = LLMReader(ask=lambda p: "")


def entry(game, level, frame):
    f = np.asarray(frame)
    bg = background_color(f)
    tr = SceneTracker(); tr.update(f, bg)
    things = list(tr.prev)
    table = [{"sid": s.sid, "colour": s.body, "size": s.size, "box": list(s.bbox), "tags": sorted(t for t in s.tags if t in ("hud", "terrain", "texture", "bar"))}
             for s in things if s.sid < 1000]
    return {"game": game, "level": level, "frame": ["".join("%x" % int(v) for v in row) for row in f],
            "things": table, "prompt": reader.prompt(f, things), "gold": gold.get(f"{game}:{level}", gold.get(game))}


arc = arc_agi.Arcade(operation_mode="offline")
rows = []
for g in PUBLIC:
    gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(g)][0]
    env = arc.make(gid)
    rows.append(entry(g, 1, np.array(env.observation_space.frame)[-1]))
    rp = ROOT / "study" / "replays" / f"{g}_L2.json"
    if rp.exists():
        d = json.load(open(rp))
        fr = d["frames"][0]["frame"]
        f = np.array([int(ch, 16) for ch in fr]).reshape(64, 64)
        rows.append(entry(g, 2, f))
    print(g, "levels", [r["level"] for r in rows if r["game"] == g], "prompt chars", len(rows[-1]["prompt"]))
json.dump(rows, open(OUT / "dataset.json", "w"))
print(f"{len(rows)} scenes -> {OUT / 'dataset.json'} ({(OUT / 'dataset.json').stat().st_size // 1024} KB)")
