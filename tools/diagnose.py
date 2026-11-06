"""Instrumented diagnosis runs: play each named game, save frame snapshots
as PNGs and a telemetry JSON describing what the agent did / didn't learn."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import arc_agi
from arcengine import GameAction, GameState

from agent.mapper_agent import MapperAgent

# ARC standard-ish 16 color palette
PAL = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (87, 60, 233), (46, 34, 10),
    (74, 34, 88), (255, 118, 118), (255, 255, 255), (100, 200, 150),
]


def save_png(frame, path, scale=4):
    g = np.asarray(frame[-1] if isinstance(frame, list) and isinstance(frame[0], list) and np.asarray(frame, dtype=object).ndim != 2 else frame, dtype=int)
    if g.ndim == 3:
        g = g[-1]
    rgb = np.zeros((*g.shape, 3), dtype=np.uint8)
    for i, col in enumerate(PAL):
        rgb[g == i] = col
    img = Image.fromarray(rgb).resize((g.shape[1] * scale, g.shape[0] * scale),
                                      Image.NEAREST)
    img.save(path)


def grid(obs):
    f = np.asarray(obs.frame[-1], dtype=int) if isinstance(obs.frame, list) else np.asarray(obs.frame, dtype=int)
    if f.ndim == 3:
        f = f[-1]
    return f


def diagnose(arc, game_id, max_actions=800, out_root="diag"):
    short = game_id[:4]
    out = Path(out_root) / short
    out.mkdir(parents=True, exist_ok=True)
    env = arc.make(game_id)
    obs = env.reset()
    agent = MapperAgent(seed=0)
    agent.on_episode_start(obs)

    save_png(obs.frame, out / "f0000_start.png")
    tel = {
        "game_id": game_id,
        "available_actions": [getattr(a, "name", a) for a in (obs.available_actions or [])],
        "win_levels": obs.win_levels,
        "deaths": 0,
        "level_ups": [],
        "changed_count": 0,
        "action_use": Counter(),
        "distinct_frames": 0,
    }
    seen_hashes = set()
    n = 0
    last_levels = obs.levels_completed
    snap_every = 200
    while n < max_actions and obs.state is not GameState.WIN:
        action = agent.act(obs)
        data = None
        if getattr(action, "is_complex", None) and action.is_complex():
            data = action.action_data.model_dump()
        prev = grid(obs)
        obs = env.step(action, data=data)
        n += 1
        g = grid(obs)
        changed = not np.array_equal(prev, g)
        agent.observe(action, obs, changed)
        tel["changed_count"] += int(changed)
        tel["action_use"][action.name] += 1
        seen_hashes.add(hash(g.tobytes()))
        if n % snap_every == 0:
            save_png(obs.frame, out / f"f{n:04d}.png")
        if obs.levels_completed > last_levels:
            tel["level_ups"].append(n)
            save_png(obs.frame, out / f"f{n:04d}_levelup.png")
            last_levels = obs.levels_completed
            agent.on_level_up(obs)
        if obs.state is GameState.GAME_OVER:
            tel["deaths"] += 1
            if tel["deaths"] <= 2:
                save_png(prev, out / f"f{n:04d}_predeath.png")
            obs = env.step(GameAction.RESET)
            n += 1
            agent.on_game_over(obs)
    save_png(obs.frame, out / f"f{n:04d}_end.png")

    # agent introspection
    m = agent.model
    tel["distinct_frames"] = len(seen_hashes)
    tel["levels_completed"] = obs.levels_completed
    tel["final_state"] = obs.state.name
    tel["action_use"] = dict(tel["action_use"])
    if m is not None:
        tel["avatar_pos"] = m.avatar_pos
        tel["motions"] = {a: {str(k): v for k, v in b.votes.items()}
                          for a, b in m.motion.items() if b.votes}
        tel["wall_colors"] = sorted(m.wall_colors()) if callable(getattr(m, "wall_colors", None)) else []
    av = agent.tracker.avatar_track()
    tel["tracker_avatar"] = None if av is None else {
        "color": av.color, "size": av.size, "control": round(av.control_score, 2)}
    tel["click_stats"] = {str(k): v for k, v in list(agent.click_stats.items())[:20]}
    tel["effect_counts"] = {str(k): v for k, v in list(agent.effect_counts.items())[:20]}
    tel["probe_failed"] = sorted(str(k) for k in agent.probe_failed_keys)[:20]
    tel["death_tail"] = {str(k): v for k, v in agent._death_tail.items()}
    (out / "telemetry.json").write_text(json.dumps(tel, indent=2, default=str))
    print(f"{short}: levels={tel['levels_completed']}/{tel['win_levels']} "
          f"deaths={tel['deaths']} changed={tel['changed_count']}/{n} "
          f"distinct_frames={tel['distinct_frames']} "
          f"motions={list(tel.get('motions', {}).keys())}")
    return tel


def main():
    targets = sys.argv[1:] or ["sk48", "cn04", "dc22", "m0r0", "sp80"]
    arc = arc_agi.Arcade(operation_mode="offline")
    for info in arc.get_environments():
        if any(info.game_id.startswith(t) for t in targets):
            try:
                diagnose(arc, info.game_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{info.game_id[:4]}: DIAG ERROR {e}")


if __name__ == "__main__":
    main()
