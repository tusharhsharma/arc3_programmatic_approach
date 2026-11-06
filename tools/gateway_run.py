"""THE RERUN, REHEARSED: the bundle plays over HTTP against a local
competition-mode gateway (tools/gateway_server.py, a separate process),
through the real `agents.agent.Agent.main()` and the Swarm's shape — every
env made first, then one thread per game, all at once — and the server's
own scorecard is read at the end.  This is the path the Kaggle rerun
takes; tools/harness_run.py is the same loop offline, in-process.

    ARC_RUN_BUDGET_S=900 .venv/bin/python tools/gateway_run.py tn36 sp80 sb26
    ARC_RUN_BUDGET_S=1800 .venv/bin/python tools/gateway_run.py all      # (every public game at once)
"""
import os, sys, time, threading, runpy, logging, subprocess, json, types
from pathlib import Path
from urllib.request import urlopen
ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path("/root/ARC-AGI-3-Agents")
PORT = int(os.environ.get("GATEWAY_PORT", "8765"))
BASE = f"http://127.0.0.1:{PORT}"

server = subprocess.Popen([sys.executable, str(ROOT / "tools" / "gateway_server.py"), str(PORT)],
                          stdout=subprocess.DEVNULL, stderr=open("/tmp/gateway_server.err", "w"))
t_wait = time.time()
while True:
    try:
        with urlopen(f"{BASE}/api/games", timeout=5) as r:
            if r.status < 500:
                break
    except Exception:
        if time.time() - t_wait > 60:
            server.kill(); raise SystemExit("the gateway did not come up")
        time.sleep(1)

# the client sees only the gateway, as on Kaggle
os.environ["ARC_BASE_URL"] = BASE
os.environ["OPERATION_MODE"] = "competition"
os.environ["ARC_API_KEY"] = "rehearsal"
sys.path = [p for p in sys.path if p not in (str(ROOT), "", str(ROOT / "tools"))]
sys.path.insert(0, str(HARNESS))
logging.basicConfig(level=logging.WARNING)
_pkg = types.ModuleType("agents"); _pkg.__path__ = [str(HARNESS / "agents")]
sys.modules["agents"] = _pkg
ns = runpy.run_path(str(ROOT / "submission" / "my_agent_core.py"), run_name="bundle")
from agents.agent import Agent
assert issubclass(ns["MyAgent"], Agent), "the bundle did not bind to the harness's Agent"
import arc_agi

arc = arc_agi.Arcade()
arc.logger.setLevel(logging.WARNING)
assert arc.operation_mode == arc_agi.OperationMode.COMPETITION, arc.operation_mode
avail = [i.game_id for i in arc.available_environments]
args = sys.argv[1:] or ["tn36"]
games = avail if args == ["all"] else [g for a in args for g in avail if g.startswith(a)]
print("GATEWAY RUN", len(games), "games:", " ".join(g[:4] for g in games), flush=True)

card = arc.open_scorecard(tags=["rehearsal"])
agents, out = [], {}
t_make = time.time()
for g in games:                                   # (the Swarm makes every env before any thread starts)
    env = arc.make(g, scorecard_id=card)
    agents.append(ns["MyAgent"](card_id=card, game_id=g, agent_name="myagent", ROOT_URL=BASE, record=False, arc_env=env))
print(f"made {len(agents)} envs in {time.time() - t_make:.1f}s", flush=True)


def play(ag):
    t0 = time.time()
    try:
        ag.main(); err = None
    except BaseException as e:
        err = repr(e)
    out[ag.game_id] = (ag.frames[-1].levels_completed, ag.action_counter, round(time.time() - t0), err)
    print(f"  done {ag.game_id[:4]} {out[ag.game_id]}", flush=True)


ts = [threading.Thread(target=play, args=(a,), daemon=True) for a in agents]
t_run = time.time()
[t.start() for t in ts]; [t.join() for t in ts]
sc = arc.close_scorecard(card)
print(f"\nGATEWAY RUN done in {time.time() - t_run:.0f}s")
for g in games:
    print(" ", g[:4], out.get(g))
if sc is not None:
    d = sc.model_dump()
    json.dump(d, open("/tmp/gateway_scorecard.json", "w"), indent=1, default=str)
    print(f"scorecard {d['card_id'][:8]}: score {d['score']:.3f}  (/tmp/gateway_scorecard.json)")
    for e in d["environments"]:
        for r in e["runs"]:
            if not r["actions"]:
                continue
            print("   ", e["id"][:4], f"score {r['score']:.2f} levels {r['levels_completed']} actions {r['actions']} resets {r['resets']} {r['state']}",
                  "level_actions", r.get("level_actions"), "baselines", r.get("level_baseline_actions"))
server.terminate()
