"""Diagnosis: press a run key on a game, read the run (agent/core/runkey),
check the law against the trace seen, and simulate the mover at offsets.

  .venv/bin/python tools/run_probe.py sp80 --prefix 2,1 --run 5 --mover-colour 9
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arc_agi
from arcengine import GameAction
from agent.perception import background_color
from agent.core.entities import read_frame, EntityTracker
from agent.core import runkey


# KNOWN WAYS UP (diagnosis only — the agent never sees these): the human
# replays' presses, level by level
WAYS = {
    "sp80": ["4,4,4,5", "c27:37,5,c25:34,4,4,1,1,1,1,c16:13,4,4,4,c35:34,c37:32,4,4,5"],
}


def reach(env, obs, game, level):
    """Play the known ways up to `level` (1-based); else the old mapper."""
    ways = WAYS.get(game)
    if ways and len(ways) >= level - 1:
        for lv in range(level - 1):
            for tok in ways[lv].split(","):
                tok = tok.strip()
                if tok.startswith("c"):
                    y, x = tok[1:].split(":")
                    act = GameAction.ACTION6; data = {"x": int(x), "y": int(y)}; act.set_data(data)
                    obs = env.step(act, data=data)
                else:
                    obs = env.step(GameAction.from_id(int(tok)))
            if obs.levels_completed < lv + 1:
                print(f"the known way did not finish level {lv + 1} (levels {obs.levels_completed})")
        return obs
    from tools.core_play import reach_level
    obs, _, _ = reach_level(env, obs, level, False)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game"); ap.add_argument("--prefix", default=""); ap.add_argument("--run", type=int, default=5)
    ap.add_argument("--level", type=int, default=1)
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode="offline")
    gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(a.game)][0]
    env = arc.make(gid)
    obs = env.reset()
    if a.level > 1:
        obs = reach(env, obs, a.game, a.level)
    for tok in a.prefix.split(","):
        if tok.strip():
            obs = env.step(GameAction.from_id(int(tok)))
    f0 = np.array(obs.frame)[-1]
    bg = background_color(f0)
    tr = EntityTracker(); tr.update(read_frame(f0, bg)); before = tr.prev
    hud_rows = {r for s in before if ("hud" in s.tags or "bar" in s.tags) for r, _ in s.cells}
    print("things:", [(s.sid, s.body, s.size, s.bbox, sorted(s.tags)) for s in before if s.sid < 1000])
    obs = env.step(GameAction.from_id(a.run))
    frames = np.array(obs.frame)
    run = runkey.read_run(frames, before, f0, hud_rows, bg)
    if run is None:
        print("no run read"); return
    print({k: (v if k != "trace" else (len(v), v[:6])) for k, v in run.items()})
    print("law holds:", runkey.law_holds(f0, run, before, bg, hud_rows))
    src = next(s for s in before if s.sid == run["source"])
    inked, touched = runkey.flow(f0, src.cells.keys(), run["direction"], bg, run["colour"], hud_rows, sink=runkey.sink_of(before))
    seen = set(run["trace"])
    print("inked", len(inked), "seen", len(seen), "common", len(inked & seen), "only-inked", sorted(inked - seen)[:10], "only-seen", sorted(seen - inked)[:10])


if __name__ == "__main__":
    main()
