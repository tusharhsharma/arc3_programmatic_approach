"""Run baseline agents on all local environments and print an RHAE-style score."""
import argparse
import json

import arc_agi

from agent.agents import RandomAgent, NoveltyAgent
from agent.runner import run_episode

AGENTS = {"random": RandomAgent, "novelty": NoveltyAgent}


def rhae(per_level_actions: dict, baselines: list, win_levels: int) -> float:
    """Approximate the official scoring: quadratic efficiency per level,
    capped at 1.15, weighted by 1-indexed level number."""
    num, den = 0.0, 0.0
    for lvl in range(1, win_levels + 1):
        w = lvl
        den += w
        used = per_level_actions.get(lvl)
        if used and lvl <= len(baselines):
            s = min((baselines[lvl - 1] / used) ** 2, 1.15)
            num += w * s
    return num / den if den else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="random", choices=AGENTS)
    p.add_argument("--max-actions", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arc = arc_agi.Arcade(operation_mode="offline")
    results = []
    for info in arc.get_environments():
        agent = AGENTS[args.agent](seed=args.seed)
        r = run_episode(arc, info.game_id, agent, max_actions=args.max_actions)
        baselines = getattr(info, "baseline_actions", None) or []
        r["rhae"] = round(rhae(r["per_level_actions"], baselines, r["win_levels"]), 4)
        results.append(r)
        print(json.dumps(r, indent=2))

    if results:
        overall = sum(r["rhae"] for r in results) / len(results)
        print(f"\nOverall RHAE-style score: {overall:.2%}")


if __name__ == "__main__":
    main()
