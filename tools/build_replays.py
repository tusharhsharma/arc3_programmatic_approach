"""Build the Core Replays viewer from study/replays/*.json:
/home/claude/replays/live/index.html (the template with the INDEX
injected) and live/data/*.json, plus files.json for the Artifact publish.

    .venv/bin/python tools/build_replays.py
"""
import json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "study" / "replays"
OUT = Path("/home/claude/replays")
LIVE = OUT / "live"
(LIVE / "data").mkdir(parents=True, exist_ok=True)
index = []
for p in sorted(SRC.glob("*.json")):
    d = json.loads(p.read_text())
    shutil.copy(p, LIVE / "data" / p.name)
    index.append({"game": d["game"], "file": p.name, "won": bool(d["won"]), "steps": int(d["steps"]), "how": d.get("how", ""), "lost_why": bool((d.get("ledger") or {}).get("report"))})
tpl = (OUT / "index.template.html").read_text()
html = tpl.replace("const INDEX = /*INDEX*/[];", "const INDEX = " + json.dumps(index) + ";")
(LIVE / "index.html").write_text(html)
files = {f"data/{e['file']}": f"data/{e['file']}" for e in index}
(OUT / "files.json").write_text(json.dumps(files))
print(f"{len(index)} replays; index at {LIVE / 'index.html'}")
