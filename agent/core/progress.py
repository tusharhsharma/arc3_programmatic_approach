"""PROGRESS — goals learned from what the presses make happen.

The crux (Tushar, 2026-10-21): the core answers "what do I control"
(the fiddle's laws) and never "what counts".  A person opening a game
watches for anything that changes that they do NOT control — a counter,
a slot filling, a thing vanishing, a colour matching — and ties each
such change to what they were doing at that moment: the thing they
control had just arrived on something, they had just clicked something,
they had just dropped something somewhere.  That tie is a rule
instance; its generalisation is the goal: do the same with every like
thing, until the score completes.

This module keeps that record and proposes those goals.  Nothing here
names a game, a colour or a place: a score is a thing that changed when
no law said it would; a cause is the relation of a controllable to
another thing on the press that changed it; a goal is the cause
repeated over every thing of the cause's kind.
"""
from collections import Counter, defaultdict
import os
_ENV_DEBUG = bool(os.environ.get("PROGRESS_DEBUG"))


def _kind(s):
    return s.body


class ProgressGoal:
    """'Do the cause again for every like thing': cause ('arrive' | 'click'
    | 'press'), the kind to do it to, the things of that kind already
    done, the score it moves."""

    def __init__(self, cause, kind, score_sid, score_kind, done, n_events, keys=()):
        self.cause = cause
        self.kind = kind
        self.score_sid = score_sid
        self.score_kind = score_kind
        self.done = set(done)
        self.n_events = n_events
        self.keys = tuple(keys)
        self.refuted = False
        self.tried = 0

    def note(self):
        what = {"arrive": "arrive on", "click": "click", "press": "press", "touch": "touch"}[self.cause]
        tgt = f"every thing of kind {self.kind}" if self.cause != "press" else (f"{self.keys[0]} over every thing of kind {self.kind}" if self.kind is not None else f"{self.keys[0]} again where it counted")
        return f"progress: {what} {tgt} — it moved S{self.score_sid} (kind {self.score_kind}) {self.n_events}x"

    def __repr__(self):
        return self.note()


class Progress:
    def __init__(self):
        self.events = []            # one per change of a non-controlled thing
        self.scores = Counter()     # score sid -> changes seen
        self.presses = 0
        self.selection = None       # (kind clicked, sid of the mark the click left) while the mark stands
        self.buttons = []           # kinds whose click left or moved a mark
        self.button_sids = set()    # the things clicked that left a mark (the buttons themselves, not their likeness elsewhere)
        self.button_rows = Counter()  # where the buttons stood when clicked (the tray's line)
        self.tellings = []          # (key, telling) — what a press showed between its frames and undid
        self.runs = []              # (key, run, frame_before, things_before) — a trace drawn and taken back (agent/core/runkey)
        self.merges = {}            # kind -> kind: two of the first, brought together by a click, became one of the second
        self.reach = 0              # how far a click has called a thing to it (chebyshev)
        self.merge_events = 0

    # ------------------------------------------------------------ observe
    def observe(self, key, click, before, after, model, tracker_events=()):
        """After a press: every thing that changed, moved, was born or
        vanished while being neither a controllable nor sent by this key's
        law, is a SCORE change; it is tied to the cause the press gave it."""
        self.presses += 1
        sel_was = None
        b_by = {s.sid: s for s in before}
        a_by = {s.sid: s for s in after}
        active = set(model.active or ())
        # controllables: the body, its parts, what the key's law moved, the selected thing
        law = {}
        try:
            law = model.fixed.moving(key) if not isinstance(key, tuple) else model.fixed.moving(key)
        except Exception:
            law = {}
        # (a KEY's law names what it sends — the body, what rides it; a CLICK's
        # law names what the click changed last time, which is the score's own
        # business: a click's law is no controllable)
        ctrl = set(active) | (set(law.keys()) if not isinstance(key, tuple) else set())
        try:
            sel = model.selected_now(before)
            if sel is not None:
                ctrl.add(sel)
        except Exception:
            pass
        for b in model.bodies():
            if b & ctrl:
                ctrl |= set(b)
        ctrl |= {s for s in a_by if s >= 1000 and s // 1000 in ctrl} | {s for s in b_by if s >= 1000 and s // 1000 in ctrl}
        # the clock and the counters are the budget, not the score
        skip = set(getattr(model, "counters", {}) or {})
        if getattr(model, "clock", None) is not None:
            skip.add(model.clock)
        # ---- what the controllables did this press (the cause candidates)
        arrived = []            # (t_before, kind)
        touched = []
        for msid in ctrl:
            ma, mb = a_by.get(msid), b_by.get(msid)
            if ma is None or mb is None or msid >= 1000:
                continue
            ma_cells = set(ma.cells); mb_cells = set(mb.cells)
            for t in before:
                if t.sid == msid or t.sid in ctrl or "hud" in t.tags or "terrain" in t.tags or t.sid >= 1000:
                    continue
                tc = set(t.cells)
                now_over = bool(ma_cells & tc); was_over = bool(mb_cells & tc)
                if now_over and not was_over:
                    arrived.append(t)
                elif not now_over and ma_cells != mb_cells:
                    # newly touching (a cell's gap)
                    tb = t.bbox; ab = ma.bbox; bb = mb.bbox
                    gap_a = max(tb[0] - ab[2], ab[0] - tb[2], tb[1] - ab[3], ab[1] - tb[3])
                    gap_b = max(tb[0] - bb[2], bb[0] - tb[2], tb[1] - bb[3], bb[1] - tb[3])
                    if gap_a <= 1 < gap_b:
                        touched.append(t)
        clicked = None
        if isinstance(key, tuple) and key[0] == "click" and click is not None:
            clicked = next((t for t in before if t.sid < 1000 and tuple(click) in t.cells), None)
        # what the body stood over or against when the key was pressed (the
        # context of a press: ACTION5 over a tile picks it up; at the zone, drops it)
        over_kind = None; place = None
        body_ = [b_by[m] for m in ctrl if m in b_by and m < 1000]
        if body_:
            me = max(body_, key=lambda x: x.size)
            place = me.bbox
            mc = set(me.cells); mb_ = me.bbox
            for t in before:
                if t.sid in ctrl or t.sid >= 1000 or "hud" in t.tags or "terrain" in t.tags or t.sid in skip:
                    continue
                tc = set(t.cells); tb = t.bbox
                gap = max(tb[0] - mb_[2], mb_[0] - tb[2], tb[1] - mb_[3], mb_[1] - tb[3])
                if mc & tc or gap <= 1:
                    over_kind = _kind(t)
                    break
        # ---- the score changes
        changes = []
        for s in after:
            if s.sid >= 1000 or s.sid in ctrl or s.sid in skip or "terrain" in s.tags:
                continue
            p = b_by.get(s.sid)
            if p is None:
                changes.append((s.sid, _kind(s), "born", s.size))
                continue
            if dict(p.cells) == dict(s.cells):
                continue
            if s.sid in law and not isinstance(key, tuple):
                continue        # sent by the key: not a score
            if set(p.cells) == set(s.cells):
                cb_, ca_ = Counter(p.cells.values()), Counter(s.cells.values())
                main_ = max(cb_, key=cb_.get)
                changes.append((s.sid, _kind(s), "recolour", ca_.get(main_, 0) - cb_.get(main_, 0)))
            elif s.size != p.size:
                changes.append((s.sid, _kind(s), "grew" if s.size > p.size else "shrank", s.size - p.size))
            else:
                changes.append((s.sid, _kind(s), "moved", 0))
        for p in before:
            if p.sid >= 1000 or p.sid in ctrl or p.sid in skip or "terrain" in p.tags:
                continue
            if p.sid not in a_by and (p.sid not in law or isinstance(key, tuple)):
                # vanished — the thing arrived on vanishing is a collect, not a score of its own
                if any(t.sid == p.sid for t in arrived):
                    changes.append((p.sid, _kind(p), "consumed", -p.size))
                else:
                    changes.append((p.sid, _kind(p), "vanished", -p.size))
        # A CLICK CALLS LIKE THINGS TO IT; TWO THAT MEET BECOME THE NEXT KIND.
        # On a click with no selection: a thing that came to the click (its
        # new place holds the click) teaches the REACH; a thing of a new kind
        # standing at the click while two or more things of one kind went
        # (vanished, or one of them grew into the new thing) is a MERGE of
        # that kind into the new one.  (su15: dots to blocks, blocks to a
        # bigger block — the legend row above names the ladder)
        if isinstance(key, tuple) and key[0] == "click" and click is not None and (self.selection is None or clicked is None):
            cr_, cc_ = int(click[0]), int(click[1])
            for s in after:
                p = b_by.get(s.sid)
                if p is None or s.sid >= 1000 or s.sid in ctrl or "hud" in s.tags or "terrain" in s.tags:
                    continue
                # (a thing that came as ITSELF — one that changed kind on the way
                # is the tracker's morph of a vanished thing into a born one, and
                # its "journey" is no measure of the reach)
                if set(p.cells) != set(s.cells) and _kind(p) == _kind(s) and p.size == s.size \
                        and (s.bbox[0] - 1 <= cr_ <= s.bbox[2] + 1 and s.bbox[1] - 1 <= cc_ <= s.bbox[3] + 1):
                    pc_, sc_ = p.centre, s.centre
                    d_came_ = max(abs(pc_[0] - cr_), abs(pc_[1] - cc_))
                    # (a call takes the nearest like thing; one "coming" from farther
                    # while a nearer stood is the tracker's naming, not a journey)
                    nearer_ = any(q.sid != p.sid and _kind(q) == _kind(p) and q.size == p.size
                                  and max(abs(q.centre[0] - cr_), abs(q.centre[1] - cc_)) < d_came_ - 1
                                  and q.sid in a_by and set(a_by[q.sid].cells) == set(q.cells) for q in before)
                    if not nearer_:
                        self.reach = max(self.reach, int(max(abs(pc_[0] - sc_[0]), abs(pc_[1] - sc_[1]))))
            gone_ = Counter()
            for p in before:
                if p.sid >= 1000 or p.sid in ctrl or "hud" in p.tags or "terrain" in p.tags:
                    continue
                if p.sid not in a_by:
                    gone_[_kind(p)] += 1
                else:
                    s = a_by[p.sid]
                    if _kind(s) != _kind(p):
                        gone_[_kind(p)] += 1          # (it grew into the new thing: the tracker's morph)
            new_ = [s for s in after if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags
                    and (s.sid not in b_by or _kind(b_by[s.sid]) != _kind(s))
                    and s.bbox[0] - 1 <= cr_ <= s.bbox[2] + 1 and s.bbox[1] - 1 <= cc_ <= s.bbox[3] + 1]
            if len(new_) == 1:
                kn_ = _kind(new_[0])
                for kd_, n_ in gone_.items():
                    if kd_ != kn_ and n_ >= 2 and all(_kind(x) != kn_ or x.sid == new_[0].sid for x in after if x.sid < 1000 and set(x.cells) & set(new_[0].cells)):
                        self.merges[kd_] = kn_
                        self.merge_events += 1
                        self.last_merge_sid = new_[0].sid
                        if _ENV_DEBUG:
                            print(f"   [progress] merge: {n_} of kind {kd_} became one of kind {kn_} at the click; reach {self.reach}")
        # A CLICK WITH A SELECTION THAT BRINGS THE SELECTED KIND ONTO THE THING
        # CLICKED IS A FILL — however the tracker tells it (the button "moved"
        # to the slot; a block of its colour born there): the slot took the
        # selection, and the selection is spent.
        filled_ = False
        sel_before = self.selection
        if clicked is not None and _ENV_DEBUG:
            print(f"   [progress] click on S{clicked.sid} k{_kind(clicked)} at {clicked.bbox} selection {self.selection} kinds after at the click {[(s_.sid, _kind(s_)) for s_ in after if set(s_.cells) & set(clicked.cells)]} near after {[(s_.sid, _kind(s_), s_.bbox, sorted(s_.tags)) for s_ in after if abs(s_.bbox[0] - clicked.bbox[0]) <= 4 and abs(s_.bbox[1] - clicked.bbox[1]) <= 4]}")
        if clicked is not None and sel_before is not None and sel_before[0] != _kind(clicked):
            tc = set(clicked.cells); cb_ = clicked.bbox
            for s_ in after:
                # (the selected kind now lies ON the thing clicked, or AROUND
                # it — a hollow tile holds the slot in its hole)
                if s_.sid < 1000 and _kind(s_) == sel_before[0] and (set(s_.cells) & tc or
                        (s_.bbox[0] <= cb_[0] and s_.bbox[1] <= cb_[1] and s_.bbox[2] >= cb_[2] and s_.bbox[3] >= cb_[3])):
                    changes.append((clicked.sid, _kind(clicked), "fill", clicked.size))
                    filled_ = True
                    break
        # A CLICK THAT LEAVES A MARK IS A SELECTION: the mark (a thing that
        # appeared or changed against the thing clicked — an outline round a
        # button, a dot beside it, a part born on it) stands for the thing
        # clicked until it is gone; a key pressed meanwhile is pressed WITH
        # that selection (sb26: click a button, confirm)
        # (a thing that vanished or changed kind under its own click was consumed
        # or transformed, not selected: what stands there is no mark of it)
        if clicked is not None and not filled_ and clicked.sid in a_by and _kind(a_by[clicked.sid]) == _kind(clicked):
            cb_ = clicked.bbox
            mark_ = None
            for sid, kd, how, d in changes:
                if kd == _kind(clicked) or how in ("vanished", "consumed"):
                    continue
                m_ = a_by.get(sid)
                if m_ is None:
                    continue
                mb_ = m_.bbox
                gap_ = max(mb_[0] - cb_[2], cb_[0] - mb_[2], mb_[1] - cb_[3], cb_[1] - mb_[3])
                if gap_ <= 1 or (set(m_.cells) & set(clicked.cells)):
                    mark_ = sid
                    break
            if mark_ is None:
                for s_ in after:
                    if s_.sid >= 1000 and s_.sid // 1000 == clicked.sid and s_.sid not in b_by:
                        mark_ = s_.sid
                        break
            if mark_ is not None:
                self.selection = (_kind(clicked), mark_)
                self.button_sids.add(clicked.sid)
                self.button_rows[clicked.bbox[0]] += 1
                if _kind(clicked) not in self.buttons:
                    self.buttons.append(_kind(clicked))
        if filled_ or (self.selection is not None and self.selection[1] not in a_by):
            sel_was = self.selection
            self.selection = None
        if not changes:
            return []
        # ---- the cause, one per press (the strongest tie first)
        if clicked is not None:
            # (a click made while a selection stands is a click WITH it: the slot
            # takes the selected colour — sb26)
            sel_k = sel_before[0] if (sel_before is not None and sel_before[0] != _kind(clicked)) else None
            cause = ("click", _kind(clicked), clicked.sid, sel_k)
        elif arrived:
            t = max(arrived, key=lambda x: x.size)
            cause = ("arrive", _kind(t), t.sid)
        elif touched:
            t = max(touched, key=lambda x: x.size)
            cause = ("touch", _kind(t), t.sid)
        else:
            sel_kind = None
            if not isinstance(key, tuple):
                if self.selection is not None:
                    sel_kind = self.selection[0]
                elif sel_was is not None:
                    sel_kind = sel_was[0]       # (the mark went with this very press: the answer was given with it)
            cause = ("press", key if not isinstance(key, tuple) else key[0], over_kind, sel_kind)
        out = []
        for sid, kd, how, delta in changes:
            ev = {"press": self.presses, "key": key, "score_sid": sid, "score_kind": kd, "how": how, "delta": delta, "cause": cause, "place": place}
            self.events.append(ev)
            self.scores[sid] += 1
            out.append(ev)
        return out

    def note_run(self, key, run, frame_before, things_before):
        if run:
            self.runs.append((key, run, frame_before, list(things_before)))

    def note_telling(self, key, telling):
        if telling:
            self.tellings.append((key, telling))

    def last_telling(self, key=None):
        for k_, t_ in reversed(self.tellings):
            if key is None or k_ == key:
                return t_
        return None

    def _shadows(self):
        """Things that change on most presses are mirrors of the control
        (a display of the body's place, the budget), not a score."""
        if self.presses < 6:
            return set()
        return {sid for sid, n in self.scores.items() if n >= 0.5 * self.presses}

    # ---------------------------------------------------------- hypotheses
    def hypotheses(self, comps, model):
        """The goals the record supports, most-moved score first: for each
        (score, cause kind) with an instance, 'do the cause to every thing
        of that kind' — the things already done excluded."""
        groups = defaultdict(list)
        shadows = self._shadows()
        for ev in self.events:
            if ev["score_sid"] in shadows:
                continue
            c = ev["cause"]
            if c[0] in ("arrive", "click", "touch"):
                groups[(ev["score_kind"], c[0], c[1])].append(ev)
            else:
                groups[(ev["score_kind"], "press", (c[1], c[2], c[3] if len(c) > 3 else None))].append(ev)
        goals = []
        kinds_present = Counter(s.body for s in comps if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags)
        for (skind, ctype, ckind), evs in groups.items():
            if ctype == "press":
                key_, over_, sel_ = ckind
                if isinstance(key_, tuple) or key_ == "click":
                    continue        # (a click on nothing in particular is no press to repeat)
                if sel_ is not None:
                    # (a key pressed with a selection: click each thing of the
                    # selected kind's family and press — the answer whose sign
                    # the score liked; see sequence goals below)
                    g = ProgressGoal("press", None, evs[-1]["score_sid"], skind, (), len(evs), keys=[key_])
                    g.with_selection = sel_
                    g.sign = sum(1 if ev["delta"] > 0 else (-1 if ev["delta"] < 0 else 0) for ev in evs)
                    goals.append(g)
                    continue
                # (a press that counted while the body stood over a thing of a
                # kind: the goal is that press over every thing of the kind —
                # ACTION5 over each tile; a press that counted nowhere in
                # particular is pressed again where it counted)
                done = {ev["cause"][2] for ev in evs} if over_ is None else set()
                g = ProgressGoal("press", over_, evs[-1]["score_sid"], skind, (), len(evs), keys=[key_])
                g.places = [ev.get("place") for ev in evs if ev.get("place")]
                goals.append(g)
                continue
            done = {ev["cause"][2] for ev in evs}
            # (a kind strewn over the board is the board: at most a dozen targets — hand-chosen)
            if kinds_present.get(ckind, 0) == 0 or kinds_present.get(ckind, 0) > 12:
                continue
            # a cause whose kind was consumed (the thing vanished on arrival) is a collect
            g = ProgressGoal("click" if ctype == "click" else "arrive", ckind, evs[-1]["score_sid"], skind, done, len(evs))
            goals.append(g)
        goals.sort(key=lambda g: (-g.n_events, g.cause != "arrive"))
        return goals

    def fills(self):
        """Clicks WITH a selection that recoloured the thing clicked: the
        slots and what they took.  [(slot kind, selection kind, score sid)]"""
        out = []
        for ev in self.events:
            c = ev["cause"]
            # (the slot clicked changed — recoloured, grown, shrunk, gone — or a
            # thing was born where it was: it took the selection)
            # (only a FILL as read at the click — the selected kind now on the
            # thing clicked; a button that took the mark when clicked with
            # another selected is a button pressed, not a slot filled)
            if c[0] == "click" and len(c) > 3 and c[3] is not None and ev["how"] == "fill":
                out.append((c[1], c[3], c[2], ev["delta"]))
        return out

    def button_kinds(self):
        """Kinds whose click left or moved a mark (a selection): the buttons."""
        return list(self.buttons)

    def describe(self):
        return [f"S{ev['score_sid']}(k{ev['score_kind']}) {ev['how']} on press {ev['press']} {ev['key']} because {ev['cause'][0]} k{ev['cause'][1]}" for ev in self.events[-8:]]

    def export(self):
        return {"events": list(self.events[-50:])}


def _components(cells):
    """4-connected components of a set of cells."""
    cells = set(cells); out = []
    while cells:
        p = cells.pop(); comp = {p}; stack = [p]
        while stack:
            r, c = stack.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in cells:
                    cells.discard(q); comp.add(q); stack.append(q)
        out.append(comp)
    return out


def _cavity(cells, bb):
    """The cells enclosed by a set of cells (its hole), within its box."""
    r0, c0, r1, c1 = bb
    if r1 - r0 < 2 or c1 - c0 < 2:
        return frozenset()
    free = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in cells}
    border = [p for p in free if p[0] in (r0, r1) or p[1] in (c0, c1)]
    seen = set(border); stack = list(border)
    while stack:
        r, c = stack.pop()
        for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if q in free and q not in seen:
                seen.add(q); stack.append(q)
    return frozenset(free - seen)


def read_telling(frames, before, f_before, hud_rows=(), tail=24):
    """WHAT A PRESS SHOWED AND UNDID.  A press that returns many frames,
    paints things on the way and ends (nearly) as it began has TOLD
    something: the things it painted, in the order it painted them (a
    run that got that far), and the place that blinked at the end (where
    it stopped).  Returns None when the press showed nothing of the kind;
    else {"painted": [sids in order of first paint], "blink": [component
    cell sets], "restored": bool, "n": frames}.
    A painted thing: the thing (a part too) whose box is the smallest to
    hold a changed component — a ring filled in is the ring's.
    Nothing here names a game: the reading is of the frames alone."""
    import numpy as np
    n = len(frames)
    if n < 6:
        return None
    H, W = f_before.shape
    hud = set(hud_rows)
    boxes = []
    for s in before:
        if s.size < 2 or "hud" in s.tags:
            continue
        # (a part of several pieces is read piece by piece, named as the
        # legend reader names them: sid + i * 1000000 in order of place)
        pieces = [s.cells.keys()] if s.sid < 1000 else sorted(_components(s.cells.keys()), key=lambda c_: min(c_))
        for i, pc in enumerate(pieces):
            rs = [r for r, _ in pc]; cs = [c for _, c in pc]
            bb = (min(rs), min(cs), max(rs), max(cs))
            cav = _cavity(set(pc), bb) if len(pc) <= 400 else frozenset()
            boxes.append((bb, s.sid + (i * 1000000 if s.sid >= 1000 else 0), (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1), cav))
    painted = []; seen_p = set()
    max_changed = 0
    for i in range(n):
        d = np.argwhere(frames[i] != f_before)
        ch = [(int(r), int(c)) for r, c in d if int(r) not in hud]
        max_changed = max(max_changed, len(ch))
        if not ch or len(ch) > 400:
            continue
        for comp in _components(ch):
            r0 = min(r for r, _ in comp); c0 = min(c for _, c in comp)
            r1 = max(r for r, _ in comp); c1 = max(c for _, c in comp)
            # (a frame FILLED IN is the frame's: a thing whose hole holds the
            # painted cells comes before a thing that merely covers them — the
            # hole may be a thing of its own, the ground showing through)
            best = None
            for (b0, b1, b2, b3), sid, area, cav in boxes:
                if b0 <= r0 and b1 <= c0 and b2 >= r1 and b3 >= c1:
                    rank = (0 if (cav and comp <= cav) else 1, area)
                    if best is None or rank < best[0]:
                        best = (rank, sid)
            if best is not None and best[1] not in seen_p:
                seen_p.add(best[1]); painted.append(best[1])
    # the blink: cells that alternate over the tail of the animation
    tail = min(tail, n)
    flips = Counter()
    for i in range(n - tail + 1, n):
        d = np.argwhere(frames[i] != frames[i - 1])
        for r, c in d:
            if int(r) not in hud:
                flips[(int(r), int(c))] += 1
    blink_cells = {p for p, k in flips.items() if k >= 3}
    blink = [c for c in _components(blink_cells) if len(c) >= 2]
    d_end = [(int(r), int(c)) for r, c in np.argwhere(frames[-1] != f_before) if int(r) not in hud]
    restored = len(d_end) <= max(4, max_changed // 10)
    if not painted and not blink:
        return None
    return {"painted": painted, "blink": blink, "restored": restored, "n": n, "max_changed": max_changed}
