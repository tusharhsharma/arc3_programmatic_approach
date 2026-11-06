"""THE PERCEPTION GAUGE: what the scene tracker names on a level's first
frame — every object with its id, colours, size, box and tags — as a
picture (the frame, each object outlined and numbered) and a table, for
a person to read against what they see.  Perception is judged here, not
play: a thing the tracker splits that a person sees as one, or glues
that a person sees as two, or a player, panel or ground it does not
mark, is a diagnosis.  No agent code changes here.

    .venv/bin/python tools/perception_report.py tn36 lf52 ...     # -> /tmp/perception/<game>.png + .txt
    .venv/bin/python tools/perception_report.py all              # the public 25
"""
import os, sys, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
logging.disable(logging.INFO)
import numpy as np
from PIL import Image, ImageDraw
import arc_agi
from agent.perception import background_color
from agent.core.scene import SceneTracker

PUBLIC = "tn36 lf52 cn04 bp35 wa30 lp85 r11l tu93 sp80 m0r0 vc33 ar25 ka59 sc25 sk48 dc22 cd82 ft09 g50t ls20 re86 s5i5 sb26 su15 tr87".split()
PAL = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170), (240, 18, 190),
       (255, 133, 27), (127, 219, 255), (135, 12, 37), (255, 255, 255), (85, 85, 85), (177, 13, 201), (0, 40, 80),
       (255, 0, 255), (60, 179, 113)]
OUT = Path("/tmp/perception"); OUT.mkdir(exist_ok=True)
S = 12                                   # pixels per cell in the picture


def report(game, arc, level_actions=()):
    gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(game)][0]
    env = arc.make(gid)
    obs = env.observation_space
    f = np.array(obs.frame)[-1]
    bg = background_color(f)
    tr = SceneTracker()
    tr.update(f, bg)
    things = [s for s in tr.prev if s.sid < 1000]
    parts = [s for s in tr.prev if s.sid >= 1000]
    H, W = f.shape
    im = Image.new("RGB", (W * S, H * S), (0, 0, 0))
    dr = ImageDraw.Draw(im)
    for r in range(H):
        for c in range(W):
            dr.rectangle([c * S, r * S, c * S + S - 1, r * S + S - 1], fill=PAL[int(f[r, c]) % 16])
    lines = [f"{gid}  {H}x{W}  background colour {bg}  objects {len(things)}  parts {len(parts)}", ""]
    for s in sorted(things, key=lambda s: (-s.size, s.sid)):
        r0, c0, r1, c1 = s.bbox
        col = (255, 255, 255) if "terrain" not in s.tags else (255, 255, 0)
        dr.rectangle([c0 * S, r0 * S, (c1 + 1) * S - 1, (r1 + 1) * S - 1], outline=col, width=1)
        dr.text((c0 * S + 1, r0 * S), str(s.sid), fill=(255, 255, 255))
        colours = sorted(set(int(v) for v in s.cells.values()))
        own = [p for p in parts if p.sid // 1000 == s.sid]
        lines.append(f"S{s.sid:<3} size {s.size:<5} box r{r0}-{r1} c{c0}-{c1}  body {s.body}  colours {colours}"
                     f"  tags {sorted(s.tags) or '-'}" + (f"  parts {[(p.sid, p.body, p.size) for p in own]}" if own else ""))
    im.save(OUT / f"{game}.png")
    (OUT / f"{game}.txt").write_text("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    args = sys.argv[1:] or ["tn36"]
    games = PUBLIC if args == ["all"] else args
    arc = arc_agi.Arcade(operation_mode="offline")
    arc.logger.setLevel(logging.WARNING)
    for g in games:
        ls = report(g, arc)
        print("\n".join(ls[:1] + ls[2:14]))
        if len(ls) > 14:
            print(f"   ... {len(ls) - 14} more")
        print()
