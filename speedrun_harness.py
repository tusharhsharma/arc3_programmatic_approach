"""Two-phase local harness: learning run -> speed-run, best-run scoring
(mirrors the official max-over-runs rule verified on the live API)."""
import argparse
import json
import time

import numpy as np

import arc_agi
from arcengine import GameAction, GameState

from agent.mapper_agent import MapperAgent
from agent.speedrun import PlaybookRecorder, SpeedRunAgent
from run import rhae

SKIP = {"bt11-fd9df0622a1a", "bt33-a7c3f9d18b4e"}


def play(arc, gid, agent, max_actions, recorder=None):
    env = arc.make(gid)
    obs = env.reset()
    agent.on_episode_start(obs)
    if recorder:
        recorder.start(obs)
    n = 0
    per_level = {}
    this_level = 0
    last = obs.levels_completed
    while n < max_actions and obs.state != GameState.WIN:
        a = agent.act(obs)
        data = a.action_data.model_dump() if a.is_complex() else None
        prev = np.array(obs.frame, dtype=object) if isinstance(obs.frame, list) else None
        prev_last = np.asarray(obs.frame[-1]) if isinstance(obs.frame, list) and obs.frame else None
        obs = env.step(a, data=data)
        n += 1
        this_level += 1
        cur_last = np.asarray(obs.frame[-1]) if isinstance(obs.frame, list) and obs.frame else None
        changed = not (prev_last is not None and cur_last is not None
                       and prev_last.shape == cur_last.shape
                       and (prev_last == cur_last).all())
        if recorder:
            recorder.record(a, data, changed, obs)
        agent.observe(a, obs, changed)
        if obs.levels_completed > last:
            per_level[obs.levels_completed] = this_level
            this_level = 0
            last = obs.levels_completed
            agent.on_level_up(obs)
        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET)
            n += 1
            this_level += 1
            if recorder:
                recorder.on_game_over()
            if hasattr(agent, "on_game_over"):
                agent.on_game_over(obs)
    return {"actions": n, "levels": obs.levels_completed,
            "win_levels": obs.win_levels, "per_level": per_level}


def two_phase(arc, info, seed=0, cap1=800, cap2=400):
    gid = info.game_id
    baselines = list(info.baseline_actions or [])

    rec = PlaybookRecorder()
    p1 = play(arc, gid, MapperAgent(seed=seed), cap1, recorder=rec)
    r1 = rhae(p1["per_level"], baselines, p1["win_levels"] or len(baselines))

    p2 = play(arc, gid, SpeedRunAgent(rec.playbook, seed=seed), cap2)
    r2 = rhae(p2["per_level"], baselines, p2["win_levels"] or len(baselines))

    return {
        "game": gid[:4],
        "learn": {"actions": p1["actions"], "levels": p1["levels"],
                  "per_level": p1["per_level"], "rhae": round(r1, 4)},
        "speed": {"actions": p2["actions"], "levels": p2["levels"],
                  "per_level": p2["per_level"], "rhae": round(r2, 4)},
        "playbook_levels": sorted(rec.playbook.keys()),
        "best_rhae": round(max(r1, r2), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--cap1", type=int, default=800)
    ap.add_argument("--cap2", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arc = arc_agi.Arcade(operation_mode="offline")
    t0 = time.time()
    results = []
    for info in arc.get_environments():
        if info.game_id in SKIP:
            continue
        if args.only and not any(info.game_id.startswith(x)
                                 for x in args.only.split(",")):
            continue
        r = two_phase(arc, info, seed=args.seed, cap1=args.cap1, cap2=args.cap2)
        results.append(r)
        print(f"{r['game']}: learn {r['learn']['levels']}lv/{r['learn']['actions']}a "
              f"(rhae {r['learn']['rhae']:.3f}) -> speed {r['speed']['levels']}lv/"
              f"{r['speed']['actions']}a per-level {r['speed']['per_level']} "
              f"(rhae {r['speed']['rhae']:.3f})  BEST {r['best_rhae']:.3f}")
    if results:
        overall = sum(r["best_rhae"] for r in results) / len(results)
        print(f"\nBEST-RUN OVERALL: {overall:.2%}  ({time.time()-t0:.0f}s)")
        with open("results/speedrun.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
