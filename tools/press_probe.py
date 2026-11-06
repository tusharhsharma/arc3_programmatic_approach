"""Press a given sequence on a game and print what each press changed (a
diagnosis tool: the frames as the core sees them).

  .venv/bin/python tools/press_probe.py sp80 2,1,1,4,4,4,5 [--level N] [--full]
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arc_agi
from arcengine import GameAction

H = "0123456789ABCDEF"


def r4(g, step=4):
    out = []
    for r in range(0, g.shape[0], step):
        out.append("%2d " % r + "".join(H[int(v)] for v in g[r, ::step]))
    return "\n".join(out)


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
    ap.add_argument("game"); ap.add_argument("presses")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--frames", action="store_true", help="print every sub-frame of multi-frame presses")
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode="offline")
    gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(a.game)][0]
    env = arc.make(gid)
    obs = env.reset()
    if a.level > 1:
        obs = reach(env, obs, a.game, a.level)
    prev = np.array(obs.frame)[-1]
    print("available", obs.available_actions)
    print(r4(prev, 2 if a.full else 4))
    for tok in a.presses.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("c"):
            y, x = tok[1:].split(":")
            act = GameAction.ACTION6; data = {"x": int(x), "y": int(y)}; act.set_data(data)
            obs = env.step(act, data=data)
        else:
            act = GameAction.from_id(int(tok))
            obs = env.step(act)
        fr = np.array(obs.frame)
        last = fr[-1]
        d = np.argwhere(last != prev)
        print(f"== {tok}: frames {len(fr)} state {obs.state} levels {obs.levels_completed} changed {len(d)}")
        if a.frames and len(fr) > 1:
            for k in range(len(fr)):
                dk = np.argwhere(fr[k] != prev)
                print(f"  -- frame {k}: {len(dk)} changed vs before")
                print(r4(fr[k], 2 if a.full else 4))
        else:
            print(r4(last, 2 if a.full else 4))
        prev = last


if __name__ == "__main__":
    main()
