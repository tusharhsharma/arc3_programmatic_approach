"""Perception survey: for every local game, capture the initial frame, run the
perception stack, probe each simple action once, and report what perception
sees. Produces survey/<game>.json + survey/<game>.png contact sheets and a
summary table. This is the ground truth for perception development."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import arc_agi
from arcengine import GameAction, GameState

from agent.perception import Perceiver, last_grid
from agent.viz import contact_sheet

SKIP = {"bt11-fd9df0622a1a", "bt33-a7c3f9d18b4e"}  # toy fixtures


def survey_game(arc, info, out_dir, probe_all=True):
    gid = info.game_id
    env = arc.make(gid)
    obs = env.reset()
    perc = Perceiver()
    logical0 = perc.logical(obs.frame)
    objs0 = perc.objects(obs.frame)

    raw0 = last_grid(obs.frame)
    rep = {
        "game_id": gid,
        "title": info.title,
        "tags": list(info.tags or []),
        "baseline_actions": list(info.baseline_actions or []),
        "win_levels": obs.win_levels,
        "available_actions": list(obs.available_actions or []),
        "raw_subframes": int(np.array(obs.frame).shape[0]) if np.array(obs.frame).ndim == 3 else 1,
        "bg_color": perc.bg,
        "cell_size": perc.cell,
        "logical_shape": list(logical0.shape),
        "colors_present": sorted(set(raw0.flatten().tolist())),
        "n_objects": len(objs0),
        "object_summary": Counter(
            (o.color, o.size) for o in objs0).most_common(12),
        "action_effects": {},
    }

    grids = [logical0.copy()]
    labels = [f"{gid[:8]} reset"]
    if probe_all:
        prev = obs.frame
        for aid in (obs.available_actions or []):
            try:
                a = GameAction.from_id(aid)
            except Exception:
                continue
            if a in (GameAction.RESET,):
                continue
            data = {"x": 32, "y": 32} if a == GameAction.ACTION6 else None
            obs = env.step(a, data=data)
            d = perc.diff(prev, obs.frame)
            rep["action_effects"][a.name] = {
                "changed_cells": d.changed_cells,
                "moved": [(c, s, list(v)) for c, s, v in d.moved][:6],
                "appeared": len(d.appeared),
                "vanished": len(d.vanished),
                "state": obs.state.name,
                "levels": obs.levels_completed,
            }
            grids.append(perc.logical(obs.frame).copy())
            labels.append(f"{a.name} d={d.changed_cells}")
            prev = obs.frame
            if obs.state in (GameState.WIN, GameState.GAME_OVER):
                break

    png = str(Path(out_dir) / f"{gid.split('-')[0]}.png")
    try:
        contact_sheet(grids, labels, png, cols=4)
        rep["png"] = png
    except Exception as e:
        rep["png_error"] = str(e)

    with open(Path(out_dir) / f"{gid.split('-')[0]}.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="survey")
    p.add_argument("--only", default=None, help="comma-separated game prefixes")
    args = p.parse_args()
    Path(args.out).mkdir(exist_ok=True)

    arc = arc_agi.Arcade(operation_mode="offline")
    rows = []
    for info in arc.get_environments():
        if info.game_id in SKIP:
            continue
        if args.only and not any(info.game_id.startswith(x)
                                 for x in args.only.split(",")):
            continue
        try:
            rep = survey_game(arc, info, args.out)
            rows.append(rep)
            print(f"{rep['game_id'][:12]:14} cell={rep['cell_size']} bg={rep['bg_color']} "
                  f"objs={rep['n_objects']:3} acts={rep['available_actions']} "
                  f"levels={rep['win_levels']} subframes={rep['raw_subframes']}")
        except Exception as e:
            print(f"{info.game_id[:14]:14} SURVEY FAILED: {type(e).__name__}: {e}")
    with open(Path(args.out) / "_summary.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n{len(rows)} games surveyed -> {args.out}/")


if __name__ == "__main__":
    main()
