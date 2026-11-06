"""THE NEW CORE, ON ITS OWN: read the frame into sprites, poke every key
and every small thing once (a fixed survey policy — the planner and
experiment selection come later), induce rules and roles, and score the
report with the gate's scorer.  Prints the induced model beside the key.

  .venv/bin/python tools/core_probe.py [--only s5i5,ka59] [--budget 40] [-v]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arc_agi
from arcengine import GameAction, GameState
from agent.perception import background_color
from agent.core.entities import read_frame, EntityTracker, describe
from agent.core.rules import RuleInducer, kind_of
from tools.mechanics_gate import score, SIM


def survey(env, obs, budget, verbose=False):
    f = np.array(obs.frame)[-1]
    bg = background_color(f)
    tr = EntityTracker()
    ind = RuleInducer()
    sprites = read_frame(f, bg)
    tr.update(sprites)
    acts = [GameAction.from_id(i) for i in obs.available_actions if i != 0]
    keys = [a for a in acts if a.name not in ("ACTION6", "RESET")]
    clickable = any(a.name == "ACTION6" for a in acts)
    plan = []
    # keys, twice round
    for _ in range(2):
        plan += [(a.name, None) for a in keys]
    # clicks: every small in-field sprite once (centre), biggest first,
    # then a few points around the biggest movable thing
    if clickable:
        small = [s for s in sprites if "terrain" not in s.tags and "bar" not in s.tags and s.size <= 120]
        small.sort(key=lambda s: -s.size)
        for s in small[:14]:
            c = s.centre
            plan.append(("ACTION6", (int(round(c[0])), int(round(c[1])))))
            if s.parts >= 2 and s.size >= 20:
                # the parts of a panel are its buttons: click each colour's centre
                by_col = {}
                for (r, cc), col in s.cells.items():
                    by_col.setdefault(col, []).append((r, cc))
                for col, cells in by_col.items():
                    if len(cells) <= s.size // 2:
                        # each connected part of this colour
                        cs = set(cells)
                        while cs:
                            start = cs.pop(); comp = {start}; st = [start]
                            while st:
                                r, cc = st.pop()
                                for q in ((r+1,cc),(r-1,cc),(r,cc+1),(r,cc-1)):
                                    if q in cs:
                                        cs.discard(q); comp.add(q); st.append(q)
                            rs = [r for r, _ in comp]; ccs = [cc for _, cc in comp]
                            plan.append(("ACTION6", (int(round(sum(rs)/len(rs))), int(round(sum(ccs)/len(ccs))))))
        plan.append(("ACTION6", (32, 32)))
    plan += [(a.name, None) for a in keys]
    steps = 0
    for name, click in plan:
        if steps >= budget or obs.state in (GameState.WIN,):
            break
        a = GameAction[name]
        before = tr.prev
        if click is not None:
            data = {"x": int(click[1]), "y": int(click[0])}
            a.set_data(data)
            obs = env.step(a, data=data)
        else:
            obs = env.step(a)
        steps += 1
        f = np.array(obs.frame)[-1]
        dead = obs.state == GameState.GAME_OVER
        cur = read_frame(f, bg, split_keys=tr.split_keys)
        events = tr.update(cur)
        ind.observe(name, click, before, cur, events, game_over=dead)
        if verbose:
            print(f"   {steps:3} {name} {click or ''} -> {events[:6]}")
        if dead:
            obs = env.step(GameAction.RESET)
            f = np.array(obs.frame)[-1]
            tr.update(read_frame(f, bg, split_keys=tr.split_keys))
        if obs.levels_completed >= 1:
            break
    return ind, tr, sprites, steps, obs.levels_completed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("-v", action="store_true")
    args = p.parse_args()
    arc = arc_agi.Arcade(operation_mode="offline")
    rows = []
    for info in arc.get_environments():
        gid = info.game_id
        if gid.startswith(SIM):
            continue
        g = gid[:4]
        if args.only and g not in args.only.split(","):
            continue
        key = ROOT / "study" / "truth" / f"{g}.json"
        if not key.exists():
            continue
        truth = json.load(open(key))
        env = arc.make(gid); obs = env.reset()
        ind, tr, sprites, steps, lv = survey(env, obs, args.budget, args.v)
        rep = ind.report(tr.prev)
        sc = score(rep, truth)
        rows.append((g, sc))
        print(f"{g} steps={steps} L1={'won' if lv >= 1 else 'no '} ent={sc['entities']:.2f} rules={sc['rules']:.2f} "
              f"| missed ents {sc['missed_entities']} | missed rules {sc['missed_rules'][:6]}")
        if args.v:
            print("  sprites:\n" + describe(sprites, 14))
            for r in rep["rules"]:
                print("   rule", r)
            for e in rep["entities"]:
                print("   ent ", e)
    n = len(rows)
    print(f"\nCORE PROBE: mean entity recall {sum(s['entities'] for _, s in rows)/max(1,n):.2f}, "
          f"mean rule recall {sum(s['rules'] for _, s in rows)/max(1,n):.2f} over {n} games")


if __name__ == "__main__":
    main()
