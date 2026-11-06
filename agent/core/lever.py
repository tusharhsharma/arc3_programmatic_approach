"""THE LEVER AND THE TWIN (statements 7 and 8 of the no-reading class,
2026-10-27; read from the human replay of g50t).

A LEVER: a thing that moves when the body stands on another and moves
back when the body leaves it.  Its open state is conditioned on that
place — a law between things, not of a key.  Read from two sightings:
the body entered X and Y moved by v; the body left X and Y moved by -v.

A REWIND: a key that puts the body back at the level's start from
wherever it stood, every time.  And its TWIN: a thing of the body's
shape that appears at the start a few presses after the rewind and
re-walks the body's record — one recorded step per press, refusals
included — and stays where the record ends.  The record is the body's
steps since the last rewind (or the level's start).  With a twin the
body can be in two places: the twin holds the lever while the body
passes the gate.

Nothing here names a game, a colour, or a coordinate.
"""
from collections import defaultdict
import os as _os
_DEBUG = bool(_os.environ.get("LEVER_DEBUG"))


def _components(cells):
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


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def _shape(cells):
    b = _bbox(cells)
    return frozenset((r - b[0], c - b[1]) for r, c in cells)


def _vec(a, b):
    """The shift from cells a to cells b when b is a shifted a; else None."""
    if len(a) != len(b):
        return None
    ba, bb = _bbox(a), _bbox(b)
    v = (bb[0] - ba[0], bb[1] - ba[1])
    if frozenset((r + v[0], c + v[1]) for r, c in a) == frozenset(b):
        return v
    return None


class LeverModel:
    def __init__(self):
        self.cands = {}         # (X sid, Y sid) -> v: the body entered X and Y moved by v (one sighting)
        self.levers = {}        # (X sid, Y sid) -> v: confirmed (the body left X and Y moved back)
        self.base = {}          # Y sid -> Y's cells when nothing stands on X (closed)
        self.pads = {}          # X sid -> X's cells (as last seen clear of the body)
        self.by_kind = {}       # (kind X, kind Y) -> v: carried across levels

    def observe(self, before, after, body_sids, kind_of, frame_before=None, frame_after=None):
        """`before`/`after`: sprites; `body_sids`: the body's sids; `kind_of`:
        sprite -> kind; the frames, when given, tell a part's move the
        tracker reads as growth.  Reads one press."""
        if not body_sids:
            return
        b_by = {s.sid: s for s in before}
        a_by = {s.sid: s for s in after}
        body_b = set()
        body_a = set()
        for sid in body_sids:
            if sid in b_by:
                body_b |= set(b_by[sid].cells)
            if sid in a_by:
                body_a |= set(a_by[sid].cells)
        if not body_b or not body_a:
            return
        # what moved this press, whole, not the body
        moved = {}
        for sid, s in a_by.items():
            # (a part of a thing — sid >= 1000, the tracker's name for a piece
            # of a compound — may be the gate: a block on a wire)
            if sid in body_sids or "hud" in s.tags or "terrain" in s.tags:
                continue
            p = b_by.get(sid)
            if p is None:
                continue
            pc, sc = set(p.cells), set(s.cells)
            v = _vec(pc, sc)
            if v is not None and v != (0, 0):
                moved[sid] = (v, frozenset(pc))
                continue
            # A PART OF IT MOVED: the cells it lost and the cells it gained are
            # one shape, shifted — a block on a wire slid along (the gate of a
            # compound thing); the base is the cells it lost
            lost, gained = pc - sc, sc - pc
            # (by the frames: the cells of the thing's colour that went out and
            # came in — a tracker that keeps a moved block's old cells as the
            # thing's reads a slide as growth)
            if frame_before is not None and frame_after is not None and len(pc) >= 4:
                try:
                    col = p.body
                    lost_f = {q for q in pc | sc if int(frame_before[q]) == col and int(frame_after[q]) != col and q not in body_a}
                    gained_f = {q for q in pc | sc if int(frame_after[q]) == col and int(frame_before[q]) != col}
                    if lost_f and gained_f:
                        lost, gained = lost_f, gained_f
                except Exception:
                    pass
            if _DEBUG and (lost or gained) and len(pc) >= 4:
                print(f"   [lever] S{sid}: {len(pc)} -> {len(sc)} cells; lost {len(lost)} {sorted(lost)[:3]} gained {len(gained)} {sorted(gained)[:3]}; body {sorted(body_a)[:2]}")
            if lost and gained:
                # (the lost cells may also hold what the body now covers, and a
                # block sliding along its wire keeps the wire's cells: a lost
                # piece shifted by v must wear the colour now, and the gained
                # piece shifted back must have worn it before)
                def wore(cells_, fr_):
                    if fr_ is None:
                        return False
                    try:
                        return all(0 <= q[0] < fr_.shape[0] and 0 <= q[1] < fr_.shape[1] and int(fr_[q]) == p.body for q in cells_)
                    except Exception:
                        return False
                for gc in _components(gained):
                    if len(gc) < 4:
                        continue
                    gb = _bbox(gc)
                    for lc in _components(lost):
                        if len(lc) < 4:
                            continue
                        lb = _bbox(lc)
                        for v in {(gb[0] - lb[0], gb[1] - lb[1]), (gb[2] - lb[2], gb[3] - lb[3])}:
                            if v == (0, 0):
                                continue
                            fwd = {(r + v[0], c + v[1]) for r, c in lc}
                            bck = {(r - v[0], c - v[1]) for r, c in gc}
                            if (frame_after is not None and wore(fwd, frame_after) and wore(bck, frame_before)) \
                                    or (frame_after is None and _vec(lc, gc) == v):
                                moved[sid] = (v, frozenset(lc))
                                break
                        if sid in moved:
                            break
                    if sid in moved:
                        break
        # the places: things the body stands on now / stood on before (a pad
        # covered whole by the body is hidden while it stands there: its
        # cells are remembered from when it was clear)
        places = {}
        for X in list(before) + list(after):
            if X.sid in body_sids or "hud" in X.tags or "terrain" in X.tags or X.size > 400:
                continue
            places.setdefault(X.sid, X)
        for X in places.values():
            xc = set(X.cells)
            if X.sid in self.pads:
                xc = self.pads[X.sid]
            on_b = bool(xc & body_b)
            on_a = bool(xc & body_a)
            if not on_b and not on_a:
                if X.sid in a_by:
                    self.pads[X.sid] = set(a_by[X.sid].cells)
                continue
            if _DEBUG and (on_a or on_b):
                print(f"   [lever] on S{X.sid}: before {on_b} after {on_a}; moved {[(y, v) for y, (v, _) in moved.items()]}")
            if on_a and not on_b:
                # (the pad is what the body stands on: of a compound thing —
                # a pad wired to its gate — only the cells under the body)
                self.pads.setdefault(X.sid, set(xc & body_a) if len(xc) > 64 else set(X.cells))
                for Y, (v, base) in moved.items():
                    if Y == X.sid and len(base) == len(X.cells):
                        continue
                    self.cands[(X.sid, Y)] = v
                    self.base.setdefault(Y, set(base))
            elif on_b and not on_a:
                for Y, (v, base) in moved.items():
                    c = self.cands.get((X.sid, Y))
                    if c is not None and c == (-v[0], -v[1]):
                        self.levers[(X.sid, Y)] = c
                        try:
                            self.by_kind[(kind_of(X), kind_of(b_by[Y]))] = c
                        except Exception:
                            pass

    def open_cells(self, Y):
        v = next((v for (x, y), v in self.levers.items() if y == Y), None)
        if v is None or Y not in self.base:
            return None
        return frozenset((r + v[0], c + v[1]) for r, c in self.base[Y])

    def gates(self):
        """[(X sid, X cells, Y sid, Y closed cells, Y open cells)]"""
        out = []
        for (X, Y), v in self.levers.items():
            if X in self.pads and Y in self.base:
                out.append((X, frozenset(self.pads[X]), Y, frozenset(self.base[Y]), self.open_cells(Y)))
        return out


class TwinModel:
    def __init__(self):
        self.start = None       # the body's cells at the level's start
        self.record = []        # the body's steps since the last rewind (vectors; (0,0) when refused)
        self.rewinds = defaultdict(int)     # key -> times it put the body back at the start from elsewhere
        self.rewind_key = None
        self.last_rewind_at = None          # press index of the last rewind
        self.replay = None      # the record the twin walks (fixed at the rewind)
        self.twin = None        # the twin's sid once seen
        self.twin_i = 0         # how many of the replay the twin has walked
        self.delay = None       # presses after the rewind before the twin's first step
        self.matched = 0        # twin steps that matched the replay
        self.confirmed = False
        self.n_press = 0
        self.twin_kind = None

    def observe(self, key, before, after, body_sids, kind_of, law_vec=None, start=None):
        """`law_vec`: the key's known step for the body (a translate), when
        it has one — a return to the start by that very step is a step.
        `start`: the body's cells at the level's start (the tracker's first
        frame), when known."""
        self.n_press += 1
        b_by = {s.sid: s for s in before}
        a_by = {s.sid: s for s in after}
        body = next((sid for sid in body_sids if sid in b_by and sid in a_by), None)
        if body is None:
            return
        cb, ca = set(b_by[body].cells), set(a_by[body].cells)
        if self.start is None:
            self.start = frozenset(start) if start else frozenset(cb)
        v = _vec(cb, ca)
        if v is None:
            # (the body's cells may be read a little differently from press
            # to press — a hole showing the ground, a pad covered: its box's
            # corner tells the step)
            bb, ba = _bbox(cb), _bbox(ca)
            v = (ba[0] - bb[0], ba[1] - bb[1])
        # A REWIND: the body is at the start again, from elsewhere, by a jump
        # no step of the key makes
        if _DEBUG and key == self.rewind_key:
            print(f"   [twin] {key}: body S{body} {_bbox(cb)} -> {_bbox(ca)}; start {_bbox(self.start)}; v {v} law {law_vec}")
        unit = min((max(abs(u[0]), abs(u[1])) for u in self.record if u != (0, 0)), default=None)
        a_step = v is not None and (v == law_vec or (unit is not None and max(abs(v[0]), abs(v[1])) <= unit and (v[0] == 0 or v[1] == 0)))
        # (a return by one step of the key's own law is a step; a return by
        # a step-sized vector under a key with no law is a rewind candidate:
        # confirmed by a second return, a jump, or the twin's appearance)
        # (a key that ever STEPPED the body — a plain move of a unit, in line —
        # is an arrow, whatever else it did once: a hazard that sends the body
        # home under an arrow is a put-back, not a rewind — ka59 L2)
        if not hasattr(self, "stepped"):
            self.stepped = defaultdict(int)
        if isinstance(key, str) and v is not None and v != (0, 0) and (v[0] == 0 or v[1] == 0) and frozenset(ca) != self.start:
            self.stepped[key] += 1
        if key == self.rewind_key or self.rewinds.get(key, 0) >= 1:
            law_vec = None                  # (a key read as a rewind has no step: its "law" is the jump home)
        if isinstance(key, str) and frozenset(ca) == self.start and frozenset(cb) != self.start and v != law_vec and self.stepped.get(key, 0) == 0:
            # (returns are counted by the place they came from: a key that
            # brings the body home from two different places, and never
            # stepped it, is a rewind; one that brings it home from the one
            # place a step away is an arrow walking back — bp35)
            if not hasattr(self, "origins"):
                self.origins = defaultdict(set)
            self.origins[key].add(_bbox(cb)[:2])
            self.rewinds[key] = len(self.origins[key])
            if self.rewinds[key] >= 2 or not a_step:
                self.rewind_key = key
            self.replay = list(self.record)
            self.record = []
            self.last_rewind_at = self.n_press
            self.last_rewind_key = key
            self.twin = None; self.twin_i = 0; self.matched = 0
            self.moves_since = 0
            self.known_at_rewind = {s.sid for s in before}      # (the twin is a thing not yet seen then)
            return
        step = v if v is not None else (0, 0)
        # (THE TWIN'S CLOCK IS THE BODY'S MOVES: a refused press is no step
        # of the record, and the twin does not walk on it — read from g50t's
        # frames: after a rewind the twin appears on the first accepted move
        # and steps once per accepted move)
        moved_now = step != (0, 0)
        if moved_now:
            self.record.append(step)
        # THE TWIN: a thing of the body's shape born at the start after a rewind
        if self.replay is not None and self.last_rewind_at is not None:
            if self.twin is None:
                # (the twin may be born under the body — the record began at
                # the start, where the body stands after the rewind — and show
                # only later: a thing of the body's shape, new, standing where
                # the replay puts it after some delay, is the twin)
                self.moves_since = getattr(self, "moves_since", 0) + (1 if moved_now else 0)
                since = self.moves_since
                sb = _bbox(self.start)
                known = getattr(self, "known_at_rewind", set())
                if _DEBUG:
                    print(f"   [twin] looking: since {since} replay {self.replay[:6]}... like-shaped {[(s.sid, s.body, s.size, s.bbox) for s in after if s.sid != body and s.sid < 1000 and abs(s.size - len(self.start)) <= 2]} start {_bbox(self.start)} shape-eq {[s.sid for s in after if s.sid != body and s.sid < 1000 and _shape(set(s.cells)) == _shape(self.start)]}")
                for s in after:
                    # (the tracker may give the new twin the old twin's name:
                    # a thing is the twin by its shape and its place, not its name)
                    if s.sid == body or s.sid >= 1000 or "hud" in s.tags:
                        continue
                    if _shape(set(s.cells)) != _shape(self.start):
                        continue
                    bb = _bbox(set(s.cells))
                    off = (bb[0] - sb[0], bb[1] - sb[1])
                    for d in range(0, 4):
                        k = since - d
                        if k < 0:
                            continue
                        walked = self.replay[:k]
                        pos = (sum(u[0] for u in walked), sum(u[1] for u in walked))
                        if pos == off and (k > 0 or frozenset(ca) != self.start):
                            self.twin = s.sid
                            self.twin_kind = kind_of(s)
                            self.delay = d
                            self.twin_i = k
                            self.matched = 1 if k > 0 else 0
                            break
                    if self.twin is not None:
                        break
            elif self.twin in b_by and self.twin in a_by and moved_now:
                tb, ta = _bbox(set(b_by[self.twin].cells)), _bbox(set(a_by[self.twin].cells))
                tv = (ta[0] - tb[0], ta[1] - tb[1])
                want = self.replay[self.twin_i] if self.twin_i < len(self.replay) else (0, 0)
                if tv == want:
                    self.matched += 1
                self.twin_i += 1
                if self.matched >= 2:
                    if self.rewind_key is None:
                        self.rewind_key = getattr(self, "last_rewind_key", None)
                    self.confirmed = self.rewind_key is not None

    def describe(self):
        return f"rewind {self.rewind_key} ({dict(self.rewinds)}), twin S{self.twin} kind {self.twin_kind} delay {self.delay} matched {self.matched} confirmed {self.confirmed}, record {len(self.record)} replay {len(self.replay) if self.replay is not None else None}"
