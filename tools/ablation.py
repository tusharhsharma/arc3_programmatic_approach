"""T1 ablation harness: measure each capability's contribution — and the
NAKED SCIENTIST's cold-start yield (all archetype skills off), which is
the best local predictor of hidden-set score (archetypes only pay when a
hidden game matches a studied family; the generic layer faces the rest).

Usage:
  python tools/ablation.py                    # full vs naked, all DEV
  python tools/ablation.py --config naked     # one config
  python tools/ablation.py --sweep            # leave-one-out per skill
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
from run import rhae

SKIP = {"bt11-fd9df0622a1a", "bt33-a7c3f9d18b4e"}
SIM_PREFIX = ("st", "sh")   # generated concept-composition games: tools/simbench.py runs them
HOLDOUT_PREFIXES = ("cd82", "sc25", "su15", "tn36", "tr87", "vc33", "wa30",
                    "g50t")
ARCHETYPES = ("docking", "carrier", "aligner", "multibody", "transform",
              "permuter", "blockmatch", "pegs", "edgemaze", "overlay", "refassembly", "actuators", "glyphs", "layoutswitch")
ALL_SWITCHES = ARCHETYPES + ("touch",)

CONFIGS = {
    "full": frozenset(),
    # the generic layer alone: perception, laws, exploration, goal-seek,
    # click spreading — no archetype may fire
    "naked": frozenset(ARCHETYPES),
    "naked+notouch": frozenset(ALL_SWITCHES),
}


def run_config(name, ablate, max_actions, seed, only=None):
    arc = arc_agi.Arcade(operation_mode="offline")
    rows = []
    for info in arc.get_environments():
        gid = info.game_id
        if (gid in SKIP or gid.startswith(HOLDOUT_PREFIXES)
                or gid.startswith(SIM_PREFIX)):
            continue
        if only and not any(gid.startswith(p) for p in only):
            continue
        ag = MapperAgent(seed=seed, ablate=ablate)
        t0 = time.time()
        r = run_episode(arc, gid, ag, max_actions=max_actions)
        baselines = getattr(info, "baseline_actions", None) or []
        r["rhae"] = round(rhae(r["per_level_actions"], baselines,
                               r["win_levels"]), 4)
        rows.append(r)
        print(f"  [{name}] {gid[:14]:14} levels={r['levels_completed']}/"
              f"{r['win_levels']} rhae={r['rhae']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    total_lv = sum(r["levels_completed"] for r in rows)
    mean_rhae = (sum(r["rhae"] for r in rows) / len(rows)) if rows else 0.0
    return {"config": name, "ablate": sorted(ablate), "levels": total_lv,
            "mean_rhae": round(mean_rhae, 4), "games": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-actions", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None, choices=list(CONFIGS))
    p.add_argument("--sweep", action="store_true",
                   help="leave-one-out: full minus each switch")
    p.add_argument("--only", default=None,
                   help="comma-separated game-id prefixes")
    args = p.parse_args()
    only = args.only.split(",") if args.only else None

    jobs = {}
    if args.sweep:
        jobs["full"] = CONFIGS["full"]
        for s in ALL_SWITCHES:
            jobs[f"no-{s}"] = frozenset({s})
    elif args.config:
        jobs[args.config] = CONFIGS[args.config]
    else:
        jobs["full"] = CONFIGS["full"]
        jobs["naked"] = CONFIGS["naked"]

    out = []
    for name, ab in jobs.items():
        print(f"== config {name} (ablate={sorted(ab)})", flush=True)
        out.append(run_config(name, ab, args.max_actions, args.seed, only))

    Path("results").mkdir(exist_ok=True)
    tag = "ablation-sweep" if args.sweep else (args.config or "ablation")
    path = f"results/{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n{'config':16} {'levels':>7} {'mean RHAE':>10}")
    for r in out:
        print(f"{r['config']:16} {r['levels']:>7} {r['mean_rhae']:>9.2%}")
    print("->", path)


if __name__ == "__main__":
    main()
