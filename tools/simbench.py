"""Sim benchmark: skill-acquisition efficiency on generated
concept-composition games (tools/simgen.py).

Reports ACTIONS-TO-FIRST-CLEAR per game — the fluid-intelligence proxy —
split into tuning vs HOLDOUT compositions. The tuning/holdout GAP is the
generalization number for the paper: how much worse the agent gets when
the same concepts appear in pairings it was never tuned on.

Usage:
  python tools/simbench.py [--max-actions 400] [--seed 0] [--naked]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arc_agi

from agent.mapper_agent import MapperAgent
from agent.runner import run_episode

NAKED = frozenset(("docking", "carrier", "aligner", "multibody",
                   "transform", "permuter", "blockmatch", "edgemaze", "overlay", "refassembly", "actuators", "glyphs", "layoutswitch"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-actions", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--naked", action="store_true",
                   help="archetypes disabled (generic layer only)")
    args = p.parse_args()

    arc = arc_agi.Arcade(operation_mode="offline")
    rows = []
    for info in arc.get_environments():
        gid = info.game_id
        if not gid.startswith(("st", "sh")):
            continue
        holdout = "sim-holdout" in (getattr(info, "tags", None) or [])
        ag = MapperAgent(seed=args.seed,
                         ablate=NAKED if args.naked else None)
        t0 = time.time()
        r = run_episode(arc, gid, ag, max_actions=args.max_actions)
        first = r["per_level_actions"].get(1)
        rows.append({
            "game": gid[:4], "holdout": holdout,
            "title": getattr(info, "title", ""),
            "levels": r["levels_completed"],
            "first_clear": first,
            "per_level": r["per_level_actions"],
        })
        print(f"  {gid[:4]} {'HOLD' if holdout else 'tune'} "
              f"levels={r['levels_completed']}/2 "
              f"first_clear={first} per_level={r['per_level_actions']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    from collections import defaultdict
    combos = defaultdict(lambda: [0, 0])
    for r in rows:
        c = combos[(r["title"], r["holdout"])]
        c[0] += 1
        c[1] += 1 if r["first_clear"] else 0
    print("\nper-combo L1 clear rate:")
    for (t, h), (n, ok) in sorted(combos.items(), key=lambda kv: kv[0][1]):
        print(f"  {'HOLD' if h else 'tune'} {t:42} {ok}/{n}")

    def summ(sel):
        cleared = [r for r in rows if r["holdout"] == sel and r["first_clear"]]
        n = sum(1 for r in rows if r["holdout"] == sel)
        lv = sum(r["levels"] for r in rows if r["holdout"] == sel)
        mean = (sum(r["first_clear"] for r in cleared) / len(cleared)
                if cleared else None)
        return n, lv, len(cleared), mean

    tn, tlv, tc, tm = summ(False)
    hn, hlv, hc, hm = summ(True)
    tag = "naked" if args.naked else "full"
    print(f"\n[{tag}] tuning : {tc}/{tn} games cleared L1, {tlv} levels, "
          f"mean first-clear {tm and round(tm, 1)}")
    print(f"[{tag}] HOLDOUT: {hc}/{hn} games cleared L1, {hlv} levels, "
          f"mean first-clear {hm and round(hm, 1)}")
    Path("results").mkdir(exist_ok=True)
    out = f"results/simbench-{tag}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
