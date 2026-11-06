"""THE UPLOAD GATE (Recalibration 2, 2026-09-06): does the agent PREDICT
the mechanics?  For every real game the agent plays level 1 (until the
level ends or the action cap), then emits a MODEL REPORT in the closed
vocabulary of study/TRUTH_SCHEMA.md; the report is scored against the
machine-readable key study/truth/<game>.json (which the agent never
sees).  Winning the level is not the gate; predicting the mechanics is.

  .venv/bin/python tools/mechanics_gate.py [--max-actions 300] [--only vc33,su15]

Scores per game: entity recall (role + colour), rule recall (action +
effect + colour of the thing acted on), goal (kind + mover colour +
target colour).  A game PASSES at entities >= 0.8, rules >= 0.8, goal
exact.  The gate is 25/25 plus a clean sim-holdout belief audit.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arc_agi
from agent.mapper_agent import MapperAgent
from agent.runner import run_episode

SIM = ("st", "sh", "sg", "sx", "hazard", "maze", "game", "bt")


# ---------------------------------------------------------------- report

def model_report(ag):
    """The current agent's beliefs, translated into the schema's words.
    This is the BASELINE extractor: it reads the cell-and-colour world
    model, the click laws, the class ledgers and the goal organ.  The
    rewritten core will emit its own report natively."""
    m = ag.model
    ents, rules = [], []
    goal = None
    if m is not None:
        if m.avatar_shape_key:
            ents.append({"role": "controlled", "colors": [int(m.avatar_shape_key[0])]})
        for color, (f, w) in m.color_stats.items():
            if w >= 1 and f == 0:
                ents.append({"role": "wall", "colors": [int(color)]})
            elif f >= 1:
                ents.append({"role": "floor", "colors": [int(color)]})
        for c in getattr(m, "pushable", ()):
            ents.append({"role": "mover", "colors": [int(c)]})
        for c in m.click_laws:
            ents.append({"role": "button", "colors": [int(c)]})
            rules.append({"action": "ACTION6", "effect": "teleport", "of": [int(m.avatar_shape_key[0])] if m.avatar_shape_key else [],
                          "when": f"click_on({int(c)})"})
        av = [int(m.avatar_shape_key[0])] if m.avatar_shape_key else []
        for name, vec in m.known_motions.items():
            eff = "slide" if m.is_slide(name) else "translate"
            rules.append({"action": name, "effect": eff, "of": av, "by": [int(vec[0]), int(vec[1])], "when": "always"})
        for name in getattr(ag, "unresolved", {}):
            pass
    for c in getattr(ag, "_lethal_colors", ()):
        ents.append({"role": "hazard", "colors": [int(c)]})
    for c in getattr(ag, "_collect_colors", ()):
        ents.append({"role": "collectible", "colors": [int(c)]})
    for (row, color) in getattr(ag, "_timer_bars", ()):
        ents.append({"role": "budget", "colors": [int(color)]})
    # the goal organ
    hyps = getattr(ag, "_goal_hyps", None) or []
    best = next((h for h in hyps if h.confirmed), None) or next((h for h in hyps if h.satisfied), None) \
        or next((h for h in hyps if not h.refuted), None)
    emap = getattr(ag, "_goal_emap", None)
    if best is not None:
        kind = {"align": "align", "cover": "cover"}[best.kind]
        if "socket" in best.note or "fits frame" in best.note:
            kind = "inside"
        tcol = None
        for h in hyps:
            pass
        # the marker's colour is in the note: "... C<col>[...]"
        import re
        mm = re.search(r" C(\d+)\[", best.note)
        if mm:
            tcol = int(mm.group(1))
        goal = {"kind": kind, "mover": [int(best.m_color)], "target": [tcol] if tcol is not None else []}
        ents.append({"role": "mover", "colors": [int(best.m_color)]})
        if tcol is not None:
            ents.append({"role": "socket" if kind == "inside" else "marker", "colors": [tcol]})
    if emap is not None:
        for key, cols in emap.effects.items():
            for color, es in cols.items():
                if any(e != (0, 0) for e in es):
                    if isinstance(key, tuple):
                        rules.append({"action": "ACTION6", "effect": "translate", "of": [int(color)],
                                      "when": "click_on" if key[0] == "click" else "click_near"})
                        if key[0] == "click":
                            ents.append({"role": "button", "colors": []})
                    else:
                        rules.append({"action": key, "effect": "translate", "of": [int(color)], "when": "always"})
    # dedupe
    seen, out_e = set(), []
    for e in ents:
        k = (e["role"], tuple(e["colors"]))
        if k not in seen:
            seen.add(k); out_e.append(e)
    seen, out_r = set(), []
    for r in rules:
        k = (r["action"], r["effect"], tuple(r["of"]), r.get("when"))
        if k not in seen:
            seen.add(k); out_r.append(r)
    return {"entities": out_e, "rules": out_r, "goal": goal}


# ----------------------------------------------------------------- score

def _cols(truth, eid):
    for e in truth["entities"]:
        if e["id"] == eid:
            return set(e["colors"])
    return set()


def score(report, truth):
    rep_e = report["entities"]; rep_r = report["rules"]
    # ENTITIES: every truth entity (except decor) by role + colour overlap
    t_ents = [e for e in truth["entities"] if e["role"] != "decor"]
    hit_e = 0
    miss_e = []
    for e in t_ents:
        ok = any(r["role"] == e["role"] and set(r["colors"]) & set(e["colors"]) for r in rep_e)
        hit_e += ok
        if not ok:
            miss_e.append(f"{e['role']}:{e['id']}")
    # RULES: (action, effect, colours of `of`)
    hit_r, miss_r = 0, []
    for r in truth["rules"]:
        of = _cols(truth, r.get("of")) if r.get("of") else set()
        ok = False
        for q in rep_r:
            if q["action"] != r["action"]:
                continue
            if q["effect"] != r["effect"]:
                continue
            if r["effect"] == "nothing" or not of or set(q["of"]) & of:
                ok = True
                break
        hit_r += ok
        if not ok:
            miss_r.append(f"{r['action']}:{r['effect']}:{r.get('of')}")
    g = truth["goal"]; rg = report.get("goal")
    goal_ok = False
    if rg is not None:
        mcol = _cols(truth, g["mover"]); tcol = _cols(truth, g["target"]) if g.get("target") else set()
        goal_ok = (rg["kind"] == g["kind"] and bool(set(rg["mover"]) & mcol)
                   and ((not tcol and not rg["target"]) or bool(set(rg["target"]) & tcol)))
    er = hit_e / max(1, len(t_ents)); rr = hit_r / max(1, len(truth["rules"]))
    return {"entities": round(er, 2), "rules": round(rr, 2), "goal": goal_ok,
            "pass": er >= 0.8 and rr >= 0.8 and goal_ok,
            "missed_entities": miss_e, "missed_rules": miss_r,
            "reported_goal": rg, "truth_goal": f"{g['kind']}({g['mover']}->{g.get('target')})"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-actions", type=int, default=300)
    p.add_argument("--only", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--core", action="store_true", help="score the new core (fiddling scientist on the scene tracker)")
    args = p.parse_args()
    arc = arc_agi.Arcade(operation_mode="offline")
    out_dir = ROOT / "study" / "gate"; out_dir.mkdir(parents=True, exist_ok=True)
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
        t0 = time.time()
        if args.core:
            # THE NEW CORE: fiddle, read, plan; its report comes from the
            # general model natively
            from tools.core_play2 import play_level, core_report
            env = arc.make(gid); obs = env.reset()
            err = None
            try:
                won_, steps_, how_, obs = play_level(env, obs, 40, (), False)
                lv = 1 if won_ else 0
            except Exception as e:
                won_, steps_, how_, lv = False, 0, "error", -1; err = repr(e)
            st = getattr(play_level, "last", {}) or {}
            try:
                rep = core_report(st["model"], st["tr"], st["hyps"], st["keys"], st.get("won_hyp"))
            except Exception as e:
                rep = {"entities": [], "rules": [], "goal": None}; err = (err or "") + " report:" + repr(e)
            sc = score(rep, truth)
            row = {"game": g, "levels": lv, "per_level": {1: steps_}, "score": sc, "report": rep, "secs": round(time.time() - t0, 1), "error": err, "how": how_}
            json.dump(row, open(out_dir / f"{g}_core.json", "w"), indent=1, default=str)
            rows.append(row)
            print(f"{g} L1={'won' if lv >= 1 else 'no '} ({steps_} acts) ent={sc['entities']:.2f} rules={sc['rules']:.2f} "
                  f"goal={'OK' if sc['goal'] else 'x '} {'PASS' if sc['pass'] else 'fail'} | truth {sc['truth_goal']} | reported {sc['reported_goal']} | missed {sc['missed_entities'][:4]} {sc['missed_rules'][:4]}"
                  + (f" | ERR {err}" if err else ""), flush=True)
            continue
        ag = MapperAgent(seed=args.seed)
        class LevelDone(Exception):
            pass
        won = {"n": None}
        orig = ag.on_level_up

        def lu(obs, _o=orig):
            # level 1 only: the report is read the moment the level ends,
            # before any per-level reset of beliefs
            won["n"] = ag._obs_i
            raise LevelDone()
        ag.on_level_up = lu
        try:
            r = run_episode(arc, gid, ag, max_actions=args.max_actions)
            err = None
        except LevelDone:
            r = {"levels_completed": 1, "per_level_actions": {1: won["n"]}}; err = None
        except Exception as e:
            r = {"levels_completed": -1, "per_level_actions": {}}; err = repr(e)
        rep = model_report(ag)
        sc = score(rep, truth)
        row = {"game": g, "levels": r["levels_completed"], "per_level": r["per_level_actions"],
               "score": sc, "report": rep, "secs": round(time.time() - t0, 1), "error": err}
        json.dump(row, open(out_dir / f"{g}.json", "w"), indent=1, default=str)
        rows.append(row)
        print(f"{g} L1={'won' if r['levels_completed'] >= 1 else 'no '} ent={sc['entities']:.2f} rules={sc['rules']:.2f} "
              f"goal={'OK' if sc['goal'] else 'x '} {'PASS' if sc['pass'] else 'fail'} | truth {sc['truth_goal']} | reported {sc['reported_goal']}",
              flush=True)
    n = len(rows); passed = sum(1 for x in rows if x["score"]["pass"])
    print(f"\nGATE: {passed}/{n} games predicted; mean entity recall "
          f"{sum(x['score']['entities'] for x in rows)/max(1,n):.2f}, mean rule recall "
          f"{sum(x['score']['rules'] for x in rows)/max(1,n):.2f}, goals right "
          f"{sum(1 for x in rows if x['score']['goal'])}/{n}")
    json.dump(rows, open(out_dir / ("_summary_core.json" if args.core else "_summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
