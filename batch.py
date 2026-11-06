"""Batch evaluation: run an agent on every local game, compute RHAE-style
scores, write results/<tag>.json with per-game and overall numbers."""
import argparse
import json
import time
from pathlib import Path

import arc_agi

from agent.agents import RandomAgent
from agent.mapper_agent import MapperAgent
from agent.runner import run_episode
from run import rhae

AGENTS = {"random": RandomAgent, "mapper": MapperAgent}
SKIP = {"bt11-fd9df0622a1a", "bt33-a7c3f9d18b4e"}
SIM_PREFIX = ("st", "sh")   # generated concept-composition games: tools/simbench.py runs them
# HOLDOUT: frozen for the one-shot generalization evaluation — never run,
# never tuned on, during development (see README / targets doc)
HOLDOUT_PREFIXES = ("cd82", "sc25", "su15", "tn36", "tr87", "vc33", "wa30", "g50t")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="mapper", choices=AGENTS)
    p.add_argument("--max-actions", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--include-holdout", action="store_true")
    args = p.parse_args()

    arc = arc_agi.Arcade(operation_mode="offline")
    results = []
    t0 = time.time()
    for info in arc.get_environments():
        if info.game_id in SKIP or info.game_id.startswith(SIM_PREFIX):
            continue
        if (not args.include_holdout
                and any(info.game_id.startswith(h) for h in HOLDOUT_PREFIXES)):
            continue
        if args.only and not any(info.game_id.startswith(x)
                                 for x in args.only.split(",")):
            continue
        agent = AGENTS[args.agent](seed=args.seed)
        t = time.time()
        try:
            r = run_episode(arc, info.game_id, agent,
                            max_actions=args.max_actions)
            r["error"] = None
        except Exception as e:
            r = {"game_id": info.game_id, "agent": args.agent, "actions": 0,
                 "levels_completed": 0, "win_levels": 0, "won": False,
                 "per_level_actions": {}, "error": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t, 1)
        baselines = list(info.baseline_actions or [])
        r["baselines"] = baselines
        r["rhae"] = round(rhae(
            {int(k): v for k, v in r["per_level_actions"].items()},
            baselines, r["win_levels"] or len(baselines)), 4)
        results.append(r)
        print(f"{info.game_id[:12]:14} levels={r['levels_completed']}/{r['win_levels']} "
              f"actions={r['actions']:4} rhae={r['rhae']:.3f} "
              f"{r['seconds']:5.1f}s {('ERR ' + r['error']) if r['error'] else ''}")

    overall = sum(r["rhae"] for r in results) / max(len(results), 1)
    total_levels = sum(r["levels_completed"] for r in results)
    max_levels = sum(r["win_levels"] or 0 for r in results)
    out = {
        "agent": args.agent, "seed": args.seed,
        "max_actions": args.max_actions,
        "overall_rhae": round(overall, 4),
        "levels": f"{total_levels}/{max_levels}",
        "wall_seconds": round(time.time() - t0, 1),
        "games": results,
    }
    Path("results").mkdir(exist_ok=True)
    tag = args.tag or f"{args.agent}-s{args.seed}"
    with open(f"results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nOVERALL RHAE: {overall:.2%}   levels {total_levels}/{max_levels}   "
          f"({time.time() - t0:.0f}s)  -> results/{tag}.json")


if __name__ == "__main__":
    main()
