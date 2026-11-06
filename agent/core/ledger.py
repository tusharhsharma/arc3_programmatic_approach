"""THE ATTEMPT LEDGER: why a level was lost, and what not to believe again.

A plan is a prediction.  The planner simulates every step before the
first press; the executor compares each step's prediction with what
appeared.  The FIRST DIVERGENCE names the step, the thing, what was
expected, what happened, what stood in the way, and the law that made
the prediction.  That is the failure report a person gives after a lost
game ("oh — that one doesn't block"), and it is actionable in one way:
the law is DOUBTED — taken out of what the next attempt relies on, to
be re-learned by observation — so the next attempt differs from this
one in exactly the belief that failed.  (Tushar: "we need actionable
failed report. Why it failed and how to not get into the same situation
again.")

Nothing here names a game, a colour or a place: a divergence is a
relation between a prediction and an observation; a doubt is a law by
kind.
"""
from collections import Counter


def divergence(step_i, action, sid, kind_of, expected_cells, observed_cells, before_cells, things_now, law=None, reason=""):
    """One record: thing `sid` was expected at `expected_cells` after
    `action`; it is at `observed_cells` (None: gone); `things_now` are
    the things standing now (what is in the way is read off them)."""
    from .planner import _bbox
    exp_b = _bbox(expected_cells) if expected_cells else None
    obs_b = _bbox(observed_cells) if observed_cells else None
    was_b = _bbox(before_cells) if before_cells else None
    if observed_cells is None:
        what = "gone"
    elif was_b is not None and obs_b[:2] == was_b[:2] and exp_b is not None and exp_b[:2] != was_b[:2]:
        what = "stayed"
    elif exp_b is not None and obs_b[:2] != exp_b[:2]:
        what = "elsewhere"
    else:
        what = "other"
    # what stood where the thing was expected (not the thing itself)
    ahead = []
    if expected_cells:
        exp = set(expected_cells)
        for t in things_now:
            if t.sid == sid or t.sid >= 1000 or "hud" in t.tags:
                continue
            if exp & set(t.cells):
                ahead.append((t.sid, kind_of(t), len(exp & set(t.cells))))
    ahead.sort(key=lambda x: -x[2])
    return {"step": step_i, "action": action, "sid": sid, "kind": kind_of_sid(sid, things_now, kind_of),
            "what": what, "expected": exp_b, "observed": obs_b, "was": was_b,
            "ahead": [(a, k) for a, k, _ in ahead[:3]], "law": law, "reason": reason}


def kind_of_sid(sid, things, kind_of):
    for t in things:
        if t.sid == sid:
            return kind_of(t)
    return None


def doubts_from(div, model):
    """What the first divergence puts in doubt, as laws by kind.  A mover
    that STAYED where it was expected to move: the kind that stood in its
    way, if it was held passable (or ground, or flat), is doubted there;
    otherwise the key's law for the mover's kind.  A mover ELSEWHERE or
    GONE: the key's law.  Returns [(law, kind or (key, kind))]."""
    out = []
    if div is None:
        return out
    k = div["action"]
    if div["what"] == "stayed":
        for a_sid, a_kind in div["ahead"]:
            if a_kind is None:
                continue
            if any(a_kind == b for (_, b) in getattr(model, "passable", ())):
                out.append(("passable", a_kind))
            if a_kind in getattr(model, "ground_ok", ()):
                out.append(("ground_ok", a_kind))
            if a_kind in getattr(model, "flat", ()):
                out.append(("flat", a_kind))
        # (something stood where the mover was sent and nothing held it
        # passable: the lesson is that thing's — it refused the mover, and
        # the observer has it as a wall of its kind — not the key's law,
        # which was never tried against open ground here)
        if not out and any(a_kind is not None for _, a_kind in div["ahead"]):
            return out
        if not out and isinstance(k, str) and div["kind"] is not None:
            out.append(("eff", (k, div["kind"])))
        if not out and isinstance(k, tuple) and k and k[0] in ("rclick", "click") and div["kind"] is not None:
            out.append(("rel", div["kind"]))          # (a click was to send the thing: the relative law of its kind)
    elif div["what"] in ("elsewhere", "gone") and isinstance(k, str) and div["kind"] is not None:
        out.append(("eff", (k, div["kind"])))
    return out


def apply_doubts(carry, doubts):
    """Take the doubted laws out of what the next attempt is seeded with
    (they are re-learned by observation), and record them."""
    if not doubts:
        return carry
    carry = dict(carry)
    rec = list(carry.get("doubts", []))
    for law, key in doubts:
        if law == "passable":
            carry["passable"] = {p for p in carry.get("passable", set()) if key not in p}
        elif law in ("ground_ok", "flat"):
            carry[law] = {kd for kd in carry.get(law, set()) if kd != key}
            # (what was observed stands in its place: the kind refused a move
            # once — the planner routes round it until one press through it
            # says otherwise: that press is the re-test)
            gr = Counter(carry.get("ground_refused", {}))
            gr[key] = max(1, gr.get(key, 0))
            carry["ground_refused"] = gr
        elif law == "eff":
            k, kd = key
            eff = {kk: dict(v) for kk, v in carry.get("eff", {}).items()}
            if k in eff:
                eff[k].pop(kd, None)
            carry["eff"] = eff
        elif law == "rel":
            carry["rel"] = {kd: v for kd, v in carry.get("rel", {}).items() if kd != key}
        if (law, key) not in rec:
            rec.append((law, key))
    carry["doubts"] = rec
    return carry


def verdicts(retests, model, things, kind_of):
    """After an attempt that opened with doubts: what the re-tests said.
    A key's law for a kind: restored if the model now holds a moving law
    for a thing of that kind under that key, dropped if the key was
    pressed and nothing of the kind moved.  A passable/ground doubt:
    restored if the kind is now held passable/ground again (a press went
    through), kept as blocking otherwise."""
    out = []
    for law, key in retests or ():
        if law == "eff":
            k, kd = key
            try:
                mv = model.fixed.moving(k)
            except Exception:
                mv = {}
            moved = any(kind_of_sid(sid, things, kind_of) == kd for sid in mv)
            pressed = getattr(model.fixed, "presses", {}).get(k, 0) > 0
            out.append((law, key, "restored: the law held on the re-test" if moved else ("dropped: pressed, nothing of the kind moved" if pressed else "untested: the key was not pressed")))
        elif law == "passable":
            back = any(key == b for (_, b) in getattr(model, "passable", ()))
            out.append((law, key, "restored: a press went through it" if back else "kept as blocking: nothing went through it"))
        elif law in ("ground_ok", "flat"):
            back = key in getattr(model, law, ())
            out.append((law, key, "restored: a press went onto it" if back else "kept as blocking: nothing went onto it"))
        elif law == "rel":
            back = key in getattr(model, "rel", {})
            out.append((law, key, "restored: a click sent it again" if back else "dropped: no click sent it"))
    return out


def report(game, level, attempt, how, reading, plan_len, divs, doubts, hyps_tried=(), keys_known=(), kinds_unasked=(), retested=(), agents=(), carry_law=None, agent_lines=()):
    """The failure report, as a person would say it, and as the ledger
    keeps it."""
    lines = [f"FAILURE REPORT {game} L{level} attempt {attempt}: {how}"
             + (f" — pursuing '{reading[:70]}'" if reading else " — no reading pursued")
             + (f", a plan of {plan_len} steps" if plan_len else "")]
    d = divs[0] if divs else None
    if d is not None:
        ahead = ", ".join(f"S{a} (kind {k})" for a, k in d["ahead"]) or "nothing"
        exp = f"at {d['expected'][:2]}" if d['expected'] else "?"
        obs = ("gone" if d["observed"] is None else f"at {d['observed'][:2]}")
        lines.append(f"  first divergence: step {d['step']} {d['action']}: S{d['sid']} (kind {d['kind']}) expected {exp}, {d['what']} ({obs}); in the way: {ahead}"
                     + (f"; the law used: {d['law']}" if d['law'] else "") + (f"; {d['reason']}" if d['reason'] else ""))
        kn_ = d.get("knew")
        if kn_:
            lines.append(f"  the plan knew: blocking {kn_.get('blocking')}, pushable {kn_.get('pushable')}, agents {kn_.get('agents')}, kinds never asked {kn_.get('unasked')}; plan of {kn_.get('plan_len')} steps from {kn_.get('plan_states')} states")
        if len(divs) > 1:
            lines.append(f"  ({len(divs) - 1} more divergences after it)")
    elif how.startswith("no_plan"):
        lines.append(f"  no plan from any reading: readings tried {len(hyps_tried)}; keys known {list(keys_known)[:6]}; kinds never asked {sorted(kinds_unasked)[:8]}")
    for law, key, verdict in retested or ():
        lines.append(f"  re-tested '{law} {key}' — {verdict}")
    # (the continuity monitor: what moves on its own, and the method held)
    if agents:
        lines.append(f"  on their own: things of kind {list(agents)} travel by their own choice — no key's law; what stands against them is in their hands, left to them")
        for al_ in agent_lines:
            lines.append(f"    the record of one: {al_}")
    if carry_law:
        lines.append(f"  the method held: {carry_law.get('key')} binds kind {sorted(carry_law.get('kinds', ()))} to the body; things of it go onto the {carry_law.get('target_kind')}-coloured thing")
    if doubts:
        lines.append("  not again: " + "; ".join(f"'{law} {key}' doubted — left out of the next attempt's laws, to be re-learned by one press" for law, key in doubts))
    elif d is not None:
        lines.append("  not again: nothing by kind to doubt — the refusal is remembered by place (the planner will not offer that step there)")
    return "\n".join(lines)
