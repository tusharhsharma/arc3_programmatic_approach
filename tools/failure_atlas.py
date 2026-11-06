"""FAILURE ATLAS — run every real game with full belief telemetry so each
failure can be placed at the stage of the loop where it happened
(perception / control / goal / mechanic / planning / exploration /
archetype), against the engine ground truth in study/truth/.
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import arc_agi
from agent.mapper_agent import MapperAgent
from agent.runner import run_episode

SIM = ("st", "sh", "sx", "mz", "bt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-actions", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None)
    args = p.parse_args()
    arc = arc_agi.Arcade(operation_mode="offline")
    out_dir = ROOT / "study" / "runs"; out_dir.mkdir(parents=True, exist_ok=True)
    for info in arc.get_environments():
        gid = info.game_id
        if gid.startswith(SIM) or gid.startswith(("hazard", "maze", "game")):
            continue
        if args.only and not any(gid.startswith(x) for x in args.only.split(",")):
            continue
        ag = MapperAgent(seed=args.seed)
        acts, deaths, lvl_marks = [], [], []
        orig_obs = ag.observe
        def wobs(action, obs, changed, _ag=ag):
            r = orig_obs(action, obs, changed)
            m = _ag.model
            acts.append({"n": _ag._obs_i, "a": action.name, "lv": obs.levels_completed,
                         "pos": list(m.avatar_pos) if (m and m.avatar_pos) else None,
                         "seek": list(_ag._seek_target) if _ag._seek_target else None,
                         "plan": len(_ag.plan)})
            return r
        ag.observe = wobs
        orig_go = ag.on_game_over
        def wgo(obs, _ag=ag):
            deaths.append(_ag._obs_i); return orig_go(obs)
        ag.on_game_over = wgo
        orig_lu = ag.on_level_up
        def wlu(obs, _ag=ag):
            m = _ag.model
            lvl_marks.append({"level": obs.levels_completed, "n": _ag._obs_i,
                              "beliefs": snapshot(_ag)})
            return orig_lu(obs)
        ag.on_level_up = wlu
        t0 = time.time()
        try:
            r = run_episode(arc, gid, ag, max_actions=args.max_actions)
            err = None
        except Exception as e:
            r = {"levels_completed": -1, "per_level_actions": {}, "actions": 0}; err = repr(e)
        row = {"game": gid[:4], "levels": r["levels_completed"], "win_levels": len(info.baseline_actions or []),
               "baseline": list(info.baseline_actions or []), "per_level": r["per_level_actions"],
               "actions": r["actions"], "secs": round(time.time() - t0, 1), "error": err,
               "deaths": deaths, "action_use": dict(Counter(a["a"] for a in acts)),
               "final": snapshot(ag), "level_marks": lvl_marks,
               "hlog": ag.hlog, "gov": {k: v for k, v in getattr(ag, "_gov_state", {}).items()},
               "anomalies": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in a.items()} for a in getattr(ag, "_anomalies", [])],
               "tail": acts[-40:], "unresolved": dict(getattr(ag, "unresolved", {}))}
        json.dump(row, open(out_dir / f"{gid[:4]}.json", "w"), indent=1, default=str)
        print(f"{gid[:4]} L={row['levels']}/{row['win_levels']} per={row['per_level']} deaths={len(deaths)} "
              f"use={row['action_use']} {row['secs']}s err={err}", flush=True)

def snapshot(ag):
    m = ag.model
    if m is None:
        return {}
    return {"motions": {a: list(v) for a, v in m.known_motions.items()},
            "slides": [a for a in m.motion if m.is_slide(a)],
            "shape_ops": dict(m.shape_op_actions), "inert": sorted(m.inert_actions),
            "walls": sorted(m.wall_colors()), "collect": sorted(ag._collect_colors),
            "lethal": sorted(ag._lethal_colors), "laws": [str(k) for k in m.laws],
            "click_laws": {str(k): v for k, v in m.click_laws.items()},
            "pushable": sorted(m.pushable), "avatar_color": m.avatar_shape_key[0] if m.avatar_shape_key else None,
            "avatar_size": len(m.avatar_shape_key[1]) if m.avatar_shape_key else None,
            "goal_refuted": sorted(getattr(ag, "_goal_refuted", [])),
            "timer": bool(getattr(ag, "_timer_active", False)),
            "playfield": list(ag.perc.playfield) if ag.perc.playfield else None}

if __name__ == "__main__":
    main()
