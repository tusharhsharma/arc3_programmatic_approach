"""BELIEF AUDIT — the atom's instruments, level 1.

A score says whether the agent won. This tool says whether the agent's
BELIEFS were true, how fast they became true, and how they misbehaved —
measured against the simulator's ground truth (the generator knows every
game's control map, movement rule, goal and hazard kinds, and colors).

Per game:
  control_recall   fraction of the 4 actions whose direction was learned
  control_prec     fraction of learned directions that are TRUE
  t_control        observe-step at which all learned directions were true
                   and >=3 actions known (None = never)
  wall_prec/rec    learned wall colors vs {WALL_C}
  goal_ok          collectible class == {GOAL_C} for collect games
                   (or no false collectible class for reach games)
  lethal_ok        lethal colors == {HAZ_C} for static hazards, else empty
  slide_ok         is_slide true iff movement == slide (per action)
  flipflops        direction re-adoptions per action (belief oscillation)
  stagnation       longest stretch of steps with no new belief adopted
  false_persist    false beliefs still held at the end (never refuted)

Aggregates over tuning and holdout separately, plus the composition gap.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simgen import CONTROLS  # noqa: E402

WALL_C, PLAYER_C, GOAL_C, HAZ_C = 9, 12, 8, 2


def unit(v):
    return (0 if v[0] == 0 else (1 if v[0] > 0 else -1),
            0 if v[1] == 0 else (1 if v[1] > 0 else -1))


def truth_dirs(control):
    # generator maps ACTIONk -> (dx, dy) in cells; agent learns (dr, dc)
    return {f"ACTION{k}": (dy, dx) for k, (dx, dy) in CONTROLS[control].items()}


def parse_title(title):
    parts = title.split("/")
    return {"control": parts[0], "movement": parts[1], "goal": parts[2],
            "hazard": parts[3], "big": len(parts) > 4}


def audit_game(arc, info, max_actions, seed):
    meta = parse_title(getattr(info, "title", ""))
    tdirs = truth_dirs(meta["control"])
    drift = (0, 1) if meta["movement"] == "drift_e" else (0, 0)
    ag = MapperAgent(seed=seed)
    snaps = []          # per observe step: belief snapshot

    orig_obs = ag.observe

    def wobs(action, obs, changed):
        r = orig_obs(action, obs, changed)
        m = ag.model
        # measure the agent's OWN decomposition (base delta after its
        # drift estimate) — an earlier version subtracted the drift a
        # second time and reported phantom diagonal beliefs (instrument
        # artifact, 2026-09-04)
        dirs = {}
        try:
            deltas = m.action_deltas()
        except Exception:
            deltas = dict(m.known_motions)
        for a, v in deltas.items():
            if v == (0, 0):
                continue
            dirs[a] = unit(v)
        snaps.append({
            "dirs": dirs,
            "walls": frozenset(m.wall_colors()),
            "collect": frozenset(ag._collect_colors),
            "lethal": frozenset(ag._lethal_colors),
            "slides": {a: m.is_slide(a) for a in m.motion},
        })
        return r
    ag.observe = wobs

    # deaths: lethal recall is only DEFINED where the agent had evidence
    # (a game with zero deaths cannot be a lethal "miss" — 2026-09-04)
    deaths = [0]
    if hasattr(ag, "on_game_over"):
        orig_go = ag.on_game_over

        def wgo(obs):
            deaths[0] += 1
            return orig_go(obs)
        ag.on_game_over = wgo

    t0 = time.time()
    r = run_episode(arc, info.game_id, ag, max_actions=max_actions)
    dt = time.time() - t0
    final = snaps[-1] if snaps else {"dirs": {}, "walls": frozenset(),
                                     "collect": frozenset(),
                                     "lethal": frozenset(), "slides": {}}

    # --- control beliefs
    learned = final["dirs"]
    correct = {a: d for a, d in learned.items() if tdirs.get(a) == d}
    control_recall = len(learned) / 4
    control_prec = (len(correct) / len(learned)) if learned else None
    t_control = None
    for i, s_ in enumerate(snaps):
        d = s_["dirs"]
        if len(d) >= 3 and all(tdirs.get(a) == u for a, u in d.items()):
            t_control = i
            break
    # --- flip-flops: per action, number of direction changes
    flips = 0
    last = {}
    for s_ in snaps:
        for a, u in s_["dirs"].items():
            if a in last and last[a] != u:
                flips += 1
            last[a] = u
    # --- walls
    walls = final["walls"]
    wall_prec = (len(walls & {WALL_C}) / len(walls)) if walls else None
    wall_rec = 1.0 if WALL_C in walls else 0.0
    # --- goal / lethal / slide beliefs
    if meta["goal"].startswith("collect"):
        goal_ok = final["collect"] == {GOAL_C}
    else:
        goal_ok = len(final["collect"]) == 0
    want_lethal = {HAZ_C} if meta["hazard"] == "static" else set()
    lethal_ok = final["lethal"] == want_lethal
    # precision: no false lethal color (always defined)
    lethal_prec_ok = not (final["lethal"] - want_lethal)
    # recall: defined only when a hazard exists AND the agent died at
    # least twice (the two-witness rule needs two deaths to condemn)
    if meta["hazard"] == "static" and deaths[0] >= 2:
        lethal_rec_ok = HAZ_C in final["lethal"]
    else:
        lethal_rec_ok = None
    want_slide = meta["movement"] == "slide"
    sl = final["slides"]
    slide_ok = (all(v == want_slide for v in sl.values()) if sl else None)
    # --- false beliefs persisting at the end
    false_persist = []
    for a, d in learned.items():
        if tdirs.get(a) != d:
            false_persist.append(f"dir:{a}={d}")
    for c in walls - {WALL_C}:
        false_persist.append(f"wall:{c}")
    for c in final["collect"] - ({GOAL_C} if meta["goal"].startswith("collect") else set()):
        false_persist.append(f"collect:{c}")
    for c in final["lethal"] - want_lethal:
        false_persist.append(f"lethal:{c}")
    # --- stagnation: longest run of steps with an unchanged belief set
    def key(s_):
        return (tuple(sorted(s_["dirs"].items())), s_["walls"], s_["collect"],
                s_["lethal"], tuple(sorted(s_["slides"].items())))
    longest = run = 0
    prev = None
    for s_ in snaps:
        k = key(s_)
        run = run + 1 if k == prev else 1
        prev = k
        longest = max(longest, run)
    return {
        "game": info.game_id[:4], "title": getattr(info, "title", ""),
        "holdout": "sim-holdout" in (getattr(info, "tags", None) or []),
        "levels": r["levels_completed"],
        "first_clear": r["per_level_actions"].get(1),
        "control_recall": round(control_recall, 2),
        "control_prec": None if control_prec is None else round(control_prec, 2),
        "t_control": t_control, "flipflops": flips,
        "wall_prec": wall_prec, "wall_rec": wall_rec,
        "goal_ok": goal_ok, "lethal_ok": lethal_ok, "slide_ok": slide_ok,
        "lethal_prec_ok": lethal_prec_ok, "lethal_rec_ok": lethal_rec_ok,
        "deaths": deaths[0],
        "false_persist": false_persist, "stagnation": longest,
        "steps": len(snaps), "secs": round(dt, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-actions", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None, help="comma-separated id prefixes")
    args = p.parse_args()
    only = args.only.split(",") if args.only else None
    arc = arc_agi.Arcade(operation_mode="offline")
    rows = []
    for info in arc.get_environments():
        gid = info.game_id
        if not gid.startswith(("st", "sh")):
            continue
        if only and not any(gid.startswith(o) for o in only):
            continue
        row = audit_game(arc, info, args.max_actions, args.seed)
        rows.append(row)
        print(f"  {row['game']} {'HOLD' if row['holdout'] else 'tune'} "
              f"L={row['levels']} ctrl={row['control_recall']}/{row['control_prec']} "
              f"t={row['t_control']} flips={row['flipflops']} "
              f"goal={'Y' if row['goal_ok'] else 'n'} "
              f"lethal={'Y' if row['lethal_ok'] else 'n'}"
              f"(p{'Y' if row['lethal_prec_ok'] else 'n'}/r{row['lethal_rec_ok']}/d{row['deaths']}) "
              f"slide={row['slide_ok']} stag={row['stagnation']} "
              f"false={row['false_persist']}", flush=True)

    def agg(sel):
        rs = [r for r in rows if r["holdout"] == sel]
        if not rs:
            return {}
        n = len(rs)
        prec = [r["control_prec"] for r in rs if r["control_prec"] is not None]
        tcs = [r["t_control"] for r in rs if r["t_control"] is not None]
        return {
            "games": n,
            "control_recall": round(sum(r["control_recall"] for r in rs) / n, 2),
            "control_prec": round(sum(prec) / len(prec), 2) if prec else None,
            "control_learned_frac": round(len(tcs) / n, 2),
            "t_control_median": sorted(tcs)[len(tcs) // 2] if tcs else None,
            "goal_ok_frac": round(sum(r["goal_ok"] for r in rs) / n, 2),
            "lethal_ok_frac": round(sum(r["lethal_ok"] for r in rs) / n, 2),
            "lethal_prec_frac": round(sum(r["lethal_prec_ok"] for r in rs) / n, 2),
            "lethal_recall_frac": (round(sum(1 for r in rs if r["lethal_rec_ok"])
                                         / len([r for r in rs if r["lethal_rec_ok"] is not None]), 2)
                                   if any(r["lethal_rec_ok"] is not None for r in rs) else None),
            "lethal_recall_n": len([r for r in rs if r["lethal_rec_ok"] is not None]),
            "slide_ok_frac": round(sum(1 for r in rs if r["slide_ok"]) / n, 2),
            "flipflops_mean": round(sum(r["flipflops"] for r in rs) / n, 1),
            "false_persist_mean": round(sum(len(r["false_persist"]) for r in rs) / n, 2),
            "stagnation_mean": round(sum(r["stagnation"] for r in rs) / n, 1),
            "L1_cleared": sum(1 for r in rs if r["first_clear"]),
        }
    out = {"tuning": agg(False), "holdout": agg(True), "games": rows}
    print("\nTUNING :", out["tuning"])
    print("HOLDOUT:", out["holdout"])
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/beliefaudit.json", "w"), indent=1)
    print("-> results/beliefaudit.json")


if __name__ == "__main__":
    main()
