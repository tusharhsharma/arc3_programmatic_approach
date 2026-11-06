import sys, time, json, importlib.util
sys.path.insert(0, "/tmp/stub")
sys.path.insert(0, "/root/arc3")
import arc_agi
from agent.runner import run_episode
spec = importlib.util.spec_from_file_location("my_agent_v11", "/tmp/v11/my_agent_v11.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["my_agent_v11"] = mod
spec.loader.exec_module(mod)
MA = mod.MapperAgent
arc = arc_agi.Arcade(operation_mode="offline")
SKIP = {"bt11-fd9df0622a1a", "bt33-a7c3f9d18b4e"}
HOLD = ("cd82", "sc25", "su15", "tn36", "tr87", "vc33", "wa30", "g50t")
rows = {}
for info in arc.get_environments():
    gid = info.game_id
    if gid in SKIP or gid.startswith(HOLD) or gid.startswith(("st", "sh")):
        continue
    ag = MA(seed=0)
    t0 = time.time()
    try:
        r = run_episode(arc, gid, ag, max_actions=800)
        rows[gid[:4]] = (r["levels_completed"], r["per_level_actions"])
        print(f"{gid[:4]} levels={r['levels_completed']} per_level={r['per_level_actions']} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"{gid[:4]} CRASH {e!r}", flush=True)
json.dump(rows, open("/tmp/v11/dev_results.json", "w"), indent=1)
