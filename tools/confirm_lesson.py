"""CONFIRM THE LESSON: was what the floor's win taught a hypothesis that
solves the level?

Tushar (2026-09-25): "the wins must teach something ... internally play
the game again to confirm if your learning was correct, to confirm the
robustness of the hypothesis derived."

Run 1, the joined agent up to level L: the floor wins L (or the core
does), and the level's export — the reading the win satisfied, the laws
of the keys and clicks, the hazards — is the lesson.  Run 2, a fresh
game: the floor alone brings the game to level L, then THE CORE ALONE
plays L seeded with the lesson.  Confirmed = the core wins the level by
itself from the lesson; a lesson that does not let the core win is a
description, not a hypothesis.  The baseline (the core alone, no
lesson) is printed beside it when asked.

    .venv/bin/python tools/confirm_lesson.py sx09:1 sp80:1 [--baseline]
"""
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import arc_agi
from arcengine import GameAction, GameState

from tools import core_play2 as cp


def reach_by_floor(env, obs, level, max_actions=3000):
    """The floor alone brings the game to the start of `level`."""
    fl = cp.Floor()
    since = 1; n = 0
    while obs.levels_completed < level - 1 and n < max_actions and obs.state != GameState.WIN:
        won, k, obs, since = fl.play(env, obs, 400, time.time() + 600, since)
        n += k
        if not won and k == 0:
            break
    return obs, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--baseline", action="store_true", help="also play the level with the core alone and no lesson")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode="offline")
    for t in a.targets:
        g, lv = t.split(":"); lv = int(lv)
        gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(g)][0]
        # run 1: the joined agent to level lv
        env = arc.make(gid); obs = env.reset()
        cp.play_game(env, obs, 40, a.v, 1, lv, (), g, floor=True, max_actions=4000, seconds=a.seconds)
        lesson = cp.play_game.carries.get(lv)
        les = [x for x in cp.play_game.lessons if x[1] == lv]
        if lesson is None:
            print(f"{g}:{lv} not won in run 1: nothing to confirm"); continue
        by = "the floor" if lesson.get("how") == "floor" else "the core"
        won_ = lesson.get("won")
        print(f"{g}:{lv} won by {by}; the lesson: reading {won_[3][:60] if won_ else None}; keys {sorted((lesson.get('eff') or {}).keys())}; hazards {sorted(cp._hazards_of(lesson))}", flush=True)
        # run 2: fresh game, the floor to the level, the core alone with the lesson
        env = arc.make(gid); obs = env.reset()
        obs, n0 = reach_by_floor(env, obs, lv)
        if obs.levels_completed < lv - 1:
            print(f"{g}:{lv} the floor could not bring the fresh game to the level"); continue
        won2, steps2, how2, obs2 = cp.play_level(env, obs, 40, (), a.v, carry=lesson)
        print(f"{g}:{lv} THE CORE ALONE WITH THE LESSON: {'WON' if won2 else 'lost'} in {steps2} actions ({how2}) -> {'CONFIRMED' if won2 else 'not confirmed'}", flush=True)
        if a.baseline:
            env = arc.make(gid); obs = env.reset()
            obs, _ = reach_by_floor(env, obs, lv)
            won3, steps3, how3, _ = cp.play_level(env, obs, 40, (), a.v, carry=None)
            print(f"{g}:{lv} the core alone, no lesson: {'WON' if won3 else 'lost'} in {steps3} actions ({how3})", flush=True)


if __name__ == "__main__":
    main()
