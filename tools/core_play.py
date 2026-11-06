"""THE NEW CORE PLAYS A LEVEL: survey (learn key models on sprites),
read a goal (the first-frame readings, on colour clusters), PLAN by
composing the induced effects with collisions, execute and verify.

  .venv/bin/python tools/core_play.py s5i5:2 vc33:2 ka59:1 [--budget 30] [-v]

game:level — earlier levels are played by the old agent to reach the
level under test (its wins there are not what is measured).
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arc_agi
from arcengine import GameAction, GameState
from agent.perception import background_color, segment, detect_playfield
from agent.mapper_agent import MapperAgent
from agent.core.entities import read_frame, read_components, EntityTracker
from agent.core.planner import KeyModel, plan as plan_keys, apply as plan_apply
from agent.goals import goal_scan


def reach_level(env, obs, level, verbose=False):
    ag = MapperAgent(seed=0); ag.on_episode_start(obs)
    n = 0
    while obs.levels_completed < level - 1 and n < 600:
        a = ag.act(obs); data = a.action_data.model_dump() if a.is_complex() else None
        prev = np.array(obs.frame); obs = env.step(a, data=data); n += 1
        ag.observe(a, obs, not np.array_equal(prev, np.array(obs.frame)))
        if obs.levels_completed > getattr(ag, "_levels", 0):
            ag.on_level_up(obs)
        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET); ag.on_game_over(obs)
    return obs, ag, n


def part_points(s):
    """Click points: the sprite's centre and the centre of each same-colour part."""
    pts = [(int(round(s.centre[0])), int(round(s.centre[1])))]
    if s.parts >= 2 and s.size >= 12:
        by_col = defaultdict(list)
        for p, col in s.cells.items():
            by_col[col].append(p)
        for col, cells in by_col.items():
            if len(cells) > s.size // 2:
                continue
            cs = set(cells)
            while cs:
                start = cs.pop(); comp = {start}; st = [start]
                while st:
                    r, c = st.pop()
                    for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if q in cs:
                            cs.discard(q); comp.add(q); st.append(q)
                rs = [r for r, _ in comp]; ccs = [c for _, c in comp]
                pts.append((int(round(sum(rs) / len(rs))), int(round(sum(ccs) / len(ccs)))))
    return pts


from collections import defaultdict


def play_level(env, obs, budget, confirmed=(), verbose=False):
    f = np.array(obs.frame)[-1]
    bg = background_color(f)
    tr = EntityTracker(); model = KeyModel()
    sprites = read_frame(f, bg)                     # sprites: for the survey's click points
    tr.update(read_components(f, bg))               # components: for effects and planning
    lv0 = obs.levels_completed
    if verbose:
        print(f"   play_level: levels_completed={lv0} state={obs.state.name}")
    acts = [GameAction.from_id(i) for i in obs.available_actions if i != 0]
    keys = [a.name for a in acts if a.name not in ("ACTION6", "RESET")]
    clickable = any(a.name == "ACTION6" for a in acts)
    # --- the goal: first-frame readings on colour clusters
    objs = segment(f, bg, mask=None)
    try:
        pf = detect_playfield(f)
    except Exception:
        pf = None
    hyps = goal_scan(objs, 64, 64, playfield=pf)
    conf_rel = {(r, b) for _, _, _, r, b in confirmed}
    hyps.sort(key=lambda h: (0 if (h.rule(), h.mover_bigger()) in conf_rel else 1, -h.prio))
    if verbose:
        for h in hyps[:4]:
            print("   hyp", h)
    steps = 0

    def step(key):
        nonlocal obs, steps, f
        before = tr.prev
        if isinstance(key, tuple):
            a = GameAction.ACTION6; data = {"x": int(key[1][1]), "y": int(key[1][0])}
            a.set_data(data); obs = env.step(a, data=data)
        else:
            obs = env.step(GameAction[key])
        steps += 1
        f = np.array(obs.frame)[-1]
        cur = read_components(f, bg)
        tr.update(cur)
        model.observe(key, before, cur)
        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET)
            f = np.array(obs.frame)[-1]
            tr.update(read_components(f, bg))
        return cur

    # --- survey
    survey = [k for k in keys] * 2
    if clickable:
        small = sorted([s for s in sprites if "terrain" not in s.tags and "bar" not in s.tags and s.size <= 160],
                       key=lambda s: -s.size)
        for s in small[:12]:
            for p in part_points(s):
                survey.append(("click", p))
    for k in survey[:budget]:
        if obs.levels_completed > lv0:
            return True, steps, "survey"
        step(k)
    if verbose:
        debug_model(model, tr.prev)
    # --- ALL AT ONCE: several movers of one kind with several markers of
    # one kind (two tips, two diamonds) is one conjunctive goal — every
    # mover at a marker of its own; the level ends only when all hold
    groups = defaultdict(list)
    for h in hyps:
        groups[(h.rule(), h.m_color)].append(h)
    conj = []
    for (rule, col), hs in groups.items():
        movers = {h.m_start for h in hs}; targets = {h.target_centre() for h in hs}
        if len(movers) >= 2 and len(targets) >= 2 and len(movers) == len(targets):
            conj.append((hs[0], sorted(movers), sorted(targets)))
    for h0, m_starts, tcs in conj:
        m_col = h0.m_color
        sids = []
        for ms in m_starts:
            cands = [s for s in tr.prev if s.body == m_col and 0.5 * h0.m_size <= s.size <= 2 * h0.m_size]
            if cands:
                sids.append(min(cands, key=lambda s: abs(s.centre[0] - ms[0]) + abs(s.centre[1] - ms[1])).sid)
        if len(set(sids)) != len(m_starts):
            continue

        def joint_dist(state, _sids=tuple(sids), _tcs=tuple(tcs)):
            # sum over movers of the distance to the nearest UNCLAIMED marker
            # (greedy assignment, nearest first)
            pos = []
            for sid in _sids:
                cs = state.get(sid)
                if not cs:
                    return 999.0
                rs = [r for r, _ in cs]; cc = [c for _, c in cs]
                pos.append(((min(rs) + max(rs)) / 2.0, (min(cc) + max(cc)) / 2.0))
            left = list(_tcs); tot = 0.0
            for p_ in pos:
                j = min(range(len(left)), key=lambda i: abs(p_[0] - left[i][0]) + abs(p_[1] - left[i][1]))
                tot += abs(p_[0] - left[j][0]) + abs(p_[1] - left[j][1]); left.pop(j)
            return tot

        tcells = set()
        solids = set()
        for s_ in tr.prev:
            if any(abs(s_.centre[0] - t[0]) + abs(s_.centre[1] - t[1]) <= 4 for t in tcs) and s_.size <= 12:
                continue
            solids.add(s_.sid)
        replans = 0
        won_here = False
        while replans < 6:
            t0 = time.time()
            p = plan_keys(model, tr.prev, lambda st: joint_dist(st) < 1.0, heuristic=lambda st: joint_dist(st) / 3.0, solids=solids)
            if verbose:
                print(f"   joint plan for {len(sids)} x {h0.rule()}: {p if p is None else len(p)} keys ({time.time()-t0:.1f}s)")
            if p is None:
                break
            diverged = False
            for i, k in enumerate(p):
                pred = {}
                for psid, e in model.moving(k).items():
                    cur_s = next((s for s in tr.prev if s.sid == psid), None)
                    if cur_s is not None:
                        pred[psid] = frozenset(plan_apply(cur_s.cells, e))
                step(k)
                if obs.levels_completed > lv0:
                    return True, steps, f"joint:{i+1}/{len(p)}", obs
                now = {s.sid: frozenset(s.cells) for s in tr.prev}
                if any(now.get(psid) is None or len(now[psid] ^ cs) > 2 for psid, cs in pred.items()):
                    diverged = True; break
            if not diverged:
                break
            replans += 1
    for h in hyps[:3]:
        m_col = h.m_color
        m_start = h.m_start
        tcen = h.target_centre()
        cands = [s for s in tr.prev if s.body == m_col and 0.5 * h.m_size <= s.size <= 2 * h.m_size]
        if not cands:
            continue
        mover = min(cands, key=lambda s: abs(s.centre[0] - m_start[0]) + abs(s.centre[1] - m_start[1]))
        sid = mover.sid
        axes = [False, False]
        for k in model.keys():
            e = model.moving(k).get(sid)
            if e and e[0] == "translate":
                axes[0] |= bool(e[1][0]); axes[1] |= bool(e[1][1])
            elif e and e[0] in ("extend", "retract"):
                axes[0] |= e[1][0] in ("top", "bottom"); axes[1] |= e[1][0] in ("left", "right")
        if not any(axes):
            if verbose:
                print(f"   {h}: no key moves the mover S{sid}")
            continue

        def mover_dist(state, _sid=sid, _tc=tcen, _axes=tuple(axes)):
            cs = state.get(_sid)
            if not cs:
                return 999.0
            rs = [r for r, _ in cs]; cc = [c for _, c in cs]
            r_now = (min(rs) + max(rs)) / 2.0; c_now = (min(cc) + max(cc)) / 2.0
            return (abs(r_now - _tc[0]) if _axes[0] else 0) + (abs(c_now - _tc[1]) if _axes[1] else 0)

        def goal_test(state):
            return mover_dist(state) < 1.0

        def heuristic(state):
            # in units of the smallest step any key moves the mover
            return mover_dist(state) / 3.0
        # THE MARKER IS NOT A WALL: the target of the reading is where the
        # mover goes, so the target's own cells do not refuse the mover
        # (a dot enters its diamond; a block fills its socket)
        tcells = set(h.target) if not isinstance(h.target, tuple) else set()
        solids = set()
        for s_ in tr.prev:
            near = abs(s_.centre[0] - tcen[0]) + abs(s_.centre[1] - tcen[1]) <= 4
            if (tcells and set(s_.cells) <= tcells) or (near and s_.size <= 12):
                continue
            solids.add(s_.sid)
        replans = 0
        while replans < 6:
            t0 = time.time()
            p = plan_keys(model, tr.prev, goal_test, heuristic=heuristic, solids=solids)
            if verbose:
                print(f"   plan for {h}: {p if p is None else len(p)} keys ({time.time()-t0:.1f}s)")
                st = getattr(plan_keys, "last_stats", {})
                print("      search:", {k: (v if not hasattr(v, "most_common") else v.most_common(4)) for k, v in st.items() if k != "best_path"}, "best path", st.get("best_path"))
                if p is None and st.get("best_path"):
                    # replay the best path in the model and explain each key's refusal there
                    state = {s_.sid: frozenset(s_.cells) for s_ in tr.prev}
                    for k in st["best_path"]:
                        for psid, e in model.moving(k).items():
                            if psid in state:
                                state[psid] = frozenset(plan_apply(state[psid], e))
                    print("      at best state, mover", sorted(state.get(sid, [])), "target", tcen)
                    for k in model.keys():
                        why = "ok"
                        new = dict(state); changed = []
                        for psid, e in model.moving(k).items():
                            if psid not in new:
                                continue
                            if e[0] == "retract" and len(new[psid]) <= model.min_size.get(psid, 0):
                                why = f"min {psid}"; break
                            cs = plan_apply(new[psid], e)
                            if any(r < 0 or c < 0 or r >= 64 or c >= 64 for r, c in cs):
                                why = f"bounds {psid}"; break
                            new[psid] = frozenset(cs); changed.append(psid)
                        if why == "ok":
                            for psid in changed:
                                for o, ocs in new.items():
                                    if o == psid or (o in changed) or frozenset((psid, o)) in model.overlap_ok:
                                        continue
                                    if new[psid] & ocs:
                                        why = f"overlap {psid} with {o} at {sorted(new[psid] & ocs)[:3]}"; break
                                if why != "ok":
                                    break
                        print("        ", k, why, "-> mover dist", round(mover_dist(new), 1) if why == "ok" else "")
            if p is None:
                break
            diverged = False
            for i, k in enumerate(p):
                # predict, act, verify: the first contradiction ends the plan
                pred = {}
                for psid, e in model.moving(k).items():
                    cur_s = next((s for s in tr.prev if s.sid == psid), None)
                    if cur_s is not None:
                        pred[psid] = frozenset(plan_apply(cur_s.cells, e))
                step(k)
                if obs.levels_completed > lv0:
                    return True, steps, f"plan:{i+1}/{len(p)}", obs
                now = {s.sid: frozenset(s.cells) for s in tr.prev}
                # the same two cells of slack as the effect reader (a hole
                # that rides the far end of a bar)
                bad = [psid for psid, cs in pred.items()
                       if now.get(psid) is None or len(now[psid] ^ cs) > 2]
                if bad:
                    if verbose:
                        print(f"   step {i+1} {k}: prediction failed for {bad[:4]}; replanning")
                        for psid in bad[:3]:
                            cs = now.get(psid)
                            print(f"      S{psid} predicted {sorted(pred[psid])[:4]}.. got {sorted(cs)[:4] if cs else None}.. sizes {len(pred[psid])}/{len(cs) if cs else 0}")
                        print("      plan so far:", p[:i+1])
                    diverged = True
                    break
                if steps > budget + 200:
                    break
            if not diverged:
                break
            replans += 1
    return obs.levels_completed > lv0, steps, "no_plan", obs

def debug_model(model, sprites):
    for k in model.keys():
        print("   key", k, {sid: e for sid, e in model.moving(k).items()}, "contra", model.contradicted.get(k))
    print("   sprites", sprites)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode="offline")
    for t in args.targets:
        g, lv = t.split(":")
        lo, hi = (int(lv.split("-")[0]), int(lv.split("-")[-1]))
        gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(g)][0]
        env = arc.make(gid); obs = env.reset()
        conf = ()
        if lo > 1:
            obs, ag, n0 = reach_level(env, obs, lo, args.v)
            conf = getattr(ag, "_goal_confirmed_kinds", ())
            if obs.levels_completed < lo - 1:
                print(f"{g}:{lo} could not reach the level (old agent stopped at {obs.levels_completed})")
                continue
        for lv in range(lo, hi + 1):
            won, steps, how, obs = play_level(env, obs, args.budget, conf, args.v)
            print(f"{g}:{lv} {'WON' if won else 'lost'} in {steps} actions ({how})", flush=True)
            if not won:
                break


if __name__ == "__main__":
    main()
