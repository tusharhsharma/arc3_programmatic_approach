"""THE OFFICIAL LOOP: play the bundle through the real `agents.agent.Agent`
base class of ARC-AGI-3-Agents (its main(): observation_space -> FrameData
-> choose_action -> take_action), offline on the local environment files,
several games at once as the swarm does.  Whatever differs between this
and tools/rehearsal.py's imitation of the loop is what the Kaggle rerun
sees and we did not.

    ARC_RUN_BUDGET_S=900 .venv/bin/python tools/harness_run.py tn36 sp80 sb26
"""
import os, sys, time, threading, runpy, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path("/root/ARC-AGI-3-Agents")
sys.path = [p for p in sys.path if p not in (str(ROOT), "", str(ROOT / "tools"))]
sys.path.insert(0, str(HARNESS))
logging.basicConfig(level=logging.WARNING)
# the package __init__ imports langgraph/langsmith templates we do not have;
# register a bare package so `agents.agent` and `agents.structs` load alone
import types
_pkg = types.ModuleType("agents"); _pkg.__path__ = [str(HARNESS / "agents")]
sys.modules["agents"] = _pkg
ns = runpy.run_path(str(ROOT / "submission" / "my_agent_core.py"), run_name="bundle")
from agents.agent import Agent
assert issubclass(ns["MyAgent"], Agent), "the bundle did not bind to the harness's Agent"
import arc_agi
games = sys.argv[1:] or ["tn36"]
arc = arc_agi.Arcade(operation_mode="offline")
out = {}


def play(game):
    gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(game)][0]
    env = arc.make(gid)
    ag = ns["MyAgent"](card_id="local", game_id=gid, agent_name="myagent", ROOT_URL="", record=False, arc_env=env)
    t0 = time.time()
    try:
        ag.main()
        err = None
    except Exception as e:
        err = repr(e)
    out[game] = (ag.frames[-1].levels_completed, ag.action_counter, round(time.time() - t0), err)


ts = [threading.Thread(target=play, args=(g,), daemon=True) for g in games]
[t.start() for t in ts]; [t.join() for t in ts]
print("\nHARNESS RUN", games)
for g in games:
    print(" ", g, out.get(g))
