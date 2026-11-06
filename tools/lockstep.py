"""M8 LOCKSTEP DRIFT: run a reference bundle and the current agent on one
game, print the first divergent action and the belief state around it.

  .venv/bin/python tools/lockstep.py mz02 [reference/contender_v29_base.py] [n_max]

Any behavior difference between versions must be attributable to a
logged belief difference at the first divergent action (design note M8).
"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/tmp/stub")
sys.path.insert(0, "/root/arc3")
import arc_agi  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402
from agent.mapper_agent import MapperAgent as CUR  # noqa: E402

GAME = sys.argv[1] if len(sys.argv) > 1 else "mz02"
REF = sys.argv[2] if len(sys.argv) > 2 else "/root/arc3/reference/contender_v29_base.py"
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 300
spec = importlib.util.spec_from_file_location("ref_agent", REF)
mod = importlib.util.module_from_spec(spec)
sys.modules["ref_agent"] = mod
spec.loader.exec_module(mod)
REFA = mod.MapperAgent
arc = arc_agi.Arcade(operation_mode="offline")
gid = next(i.game_id for i in arc.get_environments() if i.game_id.startswith(GAME))


def play(Agent, n_max):
    env = arc.make(gid)
    obs = env.reset()
    ag = Agent(seed=0)
    ag.on_episode_start(obs)
    acts, states = [], []
    lv = obs.levels_completed
    for n in range(n_max):
        a = ag.act(obs)
        dd = a.action_data.model_dump() if a.is_complex() else None
        acts.append((a.name, dd.get("x") if dd else None, dd.get("y") if dd else None))
        m = ag.model
        states.append({
            "pos": m.avatar_pos if m else None,
            "plan": list(ag.plan)[:6],
            "motions": dict(m.known_motions) if m else {},
            "walls": sorted(m.wall_colors()) if m else [],
            "laws": sorted(str(k) for k in getattr(m, "laws", {})) if m else [],
            "seek": getattr(ag, "_seek_color", None),
            "unres": dict(getattr(ag, "unresolved", {})),
            "pf": len(getattr(ag, "probe_failed_keys", ())),
            "lv": lv,
        })
        data = a.action_data.model_dump() if a.is_complex() else None
        prev = np.array(obs.frame)
        obs = env.step(a, data=data)
        ag.observe(a, obs, not np.array_equal(prev, np.array(obs.frame)))
        if obs.levels_completed > lv:
            lv = obs.levels_completed
            ag.on_level_up(obs)
        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET)
            ag.on_game_over(obs)
        if obs.state == GameState.WIN:
            break
    return acts, states, lv


a1, s1, l1 = play(REFA, NMAX)
a2, s2, l2 = play(CUR, NMAX)
print(f"ref: {len(a1)} actions, levels {l1} | cur: {len(a2)} actions, levels {l2}")
div = next((i for i in range(min(len(a1), len(a2))) if a1[i] != a2[i]), None)
print("first divergence at step:", div)
if div is not None:
    for i in range(max(0, div - 3), min(div + 3, len(a1), len(a2))):
        print(i, "ref:", a1[i], s1[i]["pos"], s1[i]["plan"], s1[i]["walls"],
              "| cur:", a2[i], s2[i]["pos"], s2[i]["plan"], s2[i]["walls"], s2[i]["laws"])
    i = div
    print("ref  beliefs:", {k: s1[i][k] for k in ("motions", "walls", "seek", "unres", "pf", "lv")})
    print("cur  beliefs:", {k: s2[i][k] for k in ("motions", "walls", "laws", "seek", "unres", "pf", "lv")})
