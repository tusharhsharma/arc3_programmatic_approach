"""THE DRESS REHEARSAL: the bundle played the way the harness plays it —
every game at once, one thread each, through MyAgent.choose_action —
with the run budget set small.  Prints, per game, the levels completed,
the actions spent and how many core turns it got.  What the sealed
gauges never measured (2026-09-26): on Kaggle the games contend for the
core's seats and the clock.

    ARC_RUN_BUDGET_S=1200 .venv/bin/python tools/rehearsal.py tn36 sp80 sb26 st70 st32 sx09 shf1 vc33
"""
import os, sys, time, threading, runpy, io, contextlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p not in (str(ROOT), "", str(ROOT / "tools"))]
ns = runpy.run_path(str(ROOT / "submission" / "my_agent_core.py"), run_name="bundle")
sys.path.insert(0, str(ROOT))
import arc_agi
from arcengine import GameState

games = sys.argv[1:] or ["tn36", "sp80"]
arc = arc_agi.Arcade(operation_mode="offline")
out = {}
lock = threading.Lock()


def play(game):
    gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(game)][0]
    env = arc.make(gid); obs = env.reset(); ag = ns["MyAgent"](); frames = [obs]; n = 0; t0 = time.time()
    while n <= ag.MAX_ACTIONS and not ag.is_done(frames, obs):
        act = ag.choose_action(frames, obs)
        data = act.action_data.model_dump() if act.is_complex() else None
        obs = env.step(act, data=data); frames.append(obs); n += 1
        if obs.state is GameState.WIN:
            break
    with lock:
        out[game] = (obs.levels_completed, n, round(time.time() - t0))


t_run = time.time()
ts = [threading.Thread(target=play, args=(g,), daemon=True) for g in games]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"\nREHEARSAL {len(games)} games in {round(time.time() - t_run)} s (budget {os.environ.get('ARC_RUN_BUDGET_S')})")
for g in games:
    lv, n, sec = out.get(g, (None, None, None))
    print(f"  {g}: levels {lv} actions {n} seconds {sec}")
