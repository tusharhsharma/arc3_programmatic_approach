"""PLANNING BY COMPOSING INDUCED EFFECTS.

The old planner searched the cell map of one walking body.  Here the
state is the set of sprites (their cells), each key has a MODEL of what
it does to which sprite — translate by v, grow by g, shrink by g —
learned from the transitions the survey recorded, and a plan is a
breadth-first search over key sequences in which every predicted state
is RENDERED and checked: a sprite that would overlap another (unless
those two have been seen overlapping) or leave the frame refuses the
move, exactly as the games do.  The same search serves a chain of
arms, a piston, a slider, a pusher and a walker: nothing here knows
which it is.  The goal is a predicate on the rendered state (a set of
marked cells arriving at a place).  Bounded: a state budget and a depth
cap; a plan is verified step by step and abandoned at the first
prediction the world contradicts.
"""
import os
_ENV = dict(os.environ)      # (the switches are read once: a lookup in os.environ is a Mapping call, and one stood in the planner's successor loop — 20 million calls, 60 s of a 300 s level on bp35 L2)
from collections import deque, defaultdict, Counter


def _shift(cells, v):
    return {(r + v[0], c + v[1]) for r, c in cells}


def _sweep(cells, v):
    """The cells a thing passes THROUGH when it moves by v (a step of six
    cells crosses five in between): the union of its shifts by 1..|v|
    along v, less where it stood.  A corridor is walked along, not
    jumped over; a wall between two floors is a wall."""
    n = max(abs(v[0]), abs(v[1]))
    if n <= 1:
        return _shift(cells, v) - set(cells)
    out = set()
    for k in range(1, n + 1):
        out |= _shift(cells, (round(v[0] * k / n), round(v[1] * k / n)))
    return out - set(cells)


_bbox_cache = {}


def _bbox(cells):
    # (a frozenset's box is asked for many thousands of times in a search:
    # remembered by the set itself; other iterables are computed each time)
    if isinstance(cells, frozenset):
        b = _bbox_cache.get(cells)
        if b is None:
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            b = (min(rs), min(cs), max(rs), max(cs))
            if len(_bbox_cache) > 300000:
                _bbox_cache.clear()
            _bbox_cache[cells] = b
        return b
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


SIDES = ("top", "bottom", "left", "right")


def _slab(cells, side, k):
    """The k rows/cols just beyond `side` of the cells' box, with the
    cross-section of the edge row/col (a bar extends by copying its end)."""
    r0, c0, r1, c1 = _bbox(cells)
    out = set()
    if side == "top":
        edge = [c for r, c in cells if r == r0]
        out = {(r0 - d, c) for d in range(1, k + 1) for c in edge}
    elif side == "bottom":
        edge = [c for r, c in cells if r == r1]
        out = {(r1 + d, c) for d in range(1, k + 1) for c in edge}
    elif side == "left":
        edge = [r for r, c in cells if c == c0]
        out = {(r, c0 - d) for d in range(1, k + 1) for r in edge}
    else:
        edge = [r for r, c in cells if c == c1]
        out = {(r, c1 + d) for d in range(1, k + 1) for r in edge}
    return out


def _cut(cells, side, k):
    """Remove the k rows/cols at `side`."""
    r0, c0, r1, c1 = _bbox(cells)
    if side == "top":
        return {p for p in cells if p[0] > r0 + k - 1}
    if side == "bottom":
        return {p for p in cells if p[0] < r1 - k + 1}
    if side == "left":
        return {p for p in cells if p[1] > c0 + k - 1}
    return {p for p in cells if p[1] < c1 - k + 1}


def _filled(s):
    """A thing's cells with its enclosed holes filled: a ring whose dot
    changes side is still the ring, translated."""
    cs = set(s.cells)
    import os
    if len(cs) <= 400 and not _ENV.get("NOFILL"):     # (a dot showing through a mover is a hole, not a change of shape)
        try:
            cs |= set(s.cavity())
        except Exception:
            pass
    return cs


def _rot(cells, k, riders=()):
    """The cells turned k quarter-turns clockwise about their box's corner
    (rebased to the same top-left).  `riders`: other cell sets turned by
    the same map (a part on the thing); returned after the cells."""
    r0, c0, r1, c1 = _bbox(cells)
    out = set(cells); rs = [set(x) for x in riders]
    for _ in range(k % 4):
        rr0, cc0, rr1, cc1 = _bbox(out)
        h = rr1 - rr0
        f = lambda r, c: (c - cc0 + rr0, h - (r - rr0) + cc0)
        out = {f(r, c) for r, c in out}
        rs = [{f(r, c) for r, c in x} for x in rs]
    b2 = _bbox(out)
    v = (r0 - b2[0], c0 - b2[1])
    if riders:
        return _shift(out, v), [_shift(x, v) for x in rs]
    return _shift(out, v)


def effect_of(before, after):
    """What happened to one sprite between two frames, as a cell-level
    transform: ('translate', v) | ('extend', (side, k)) | ('retract',
    (side, k)) | ('rotate', k) | ('none', ..) | ('morph', ..)."""
    b, a = _filled(before), _filled(after)
    if b == a:
        return ("none", (0, 0))
    bb, ab = _bbox(b), _bbox(a)
    v = (ab[0] - bb[0], ab[1] - bb[1])
    if len(a) == len(b) and _shift(b, v) == a:
        return ("translate", v)
    if len(a) == len(b) and len(a) >= 4:
        # A TURN: the same shape a quarter-turn on (asymmetric enough that
        # a turn is not also a translation), wherever it came to rest
        # (where it comes to rest is the clamp's business, not the law's:
        # the turn is recorded alone)
        for k in (1, 3, 2):
            rb = _rot(b, k)
            if rb != b and _shift(rb, v) == a:
                return ("rotate", k)
    if len(a) > len(b):
        # extended at exactly one side?  (a hole riding the far end — a
        # tip inside a bar — may move: two cells of slack)
        for side, k in (("top", bb[0] - ab[0]), ("bottom", ab[2] - bb[2]), ("left", bb[1] - ab[1]), ("right", ab[3] - bb[3])):
            if k > 0 and len((b | _slab(b, side, k)) ^ a) <= 2:
                return ("extend", (side, k))
        return ("morph", v)
    if len(a) < len(b):
        for side, k in (("top", ab[0] - bb[0]), ("bottom", bb[2] - ab[2]), ("left", ab[1] - bb[1]), ("right", bb[3] - ab[3])):
            if k > 0 and len(_cut(b, side, k) ^ a) <= 2:
                return ("retract", (side, k))
        return ("morph", v)
    return ("morph", v)


def apply(cells, eff):
    kind, g = eff
    if kind == "translate":
        return _shift(cells, g)
    if kind == "extend":
        return set(cells) | _slab(cells, g[0], g[1])
    if kind == "retract":
        return _cut(cells, g[0], g[1])       # (cut to nothing: an emptied thing, refused further)
    if kind == "rotate":
        return _rot(cells, g)
    return set(cells)


def _same_way(a, b):
    """Two moves along one axis, the same way."""
    if (a[0] == 0) != (b[0] == 0) or (a[1] == 0) != (b[1] == 0):
        return False
    if a[0] and b[0]:
        return (a[0] > 0) == (b[0] > 0) and a[1] == 0
    if a[1] and b[1]:
        return (a[1] > 0) == (b[1] > 0) and a[0] == 0
    return False


def _unit_step(a, b, extent=None):
    """The step of a slide: what divides every move seen and the thing's
    own extent along the axis (a board with a pitch: two slides of 12 and
    24 by a thing 4 wide step by 4, not 12)."""
    import math
    la, lb = abs(a[0]) + abs(a[1]), abs(b[0]) + abs(b[1])
    g = math.gcd(la, lb) or min(la, lb)
    if extent:
        g = math.gcd(g, int(extent)) or g
    d = ((a[0] > 0) - (a[0] < 0), (a[1] > 0) - (a[1] < 0))
    return (d[0] * g, d[1] * g)


class KeyModel:
    """key -> {sid: effect}, with support and a contradiction flag."""

    def __init__(self):
        self.eff = defaultdict(dict)         # key -> sid -> effect
        self.support = defaultdict(lambda: defaultdict(int))
        self.contradicted = defaultdict(set)
        self.overlap_ok = set()               # frozenset({sid, sid}) seen overlapping legally
        self.presses = defaultdict(int)
        self.alt = defaultdict(lambda: defaultdict(Counter))   # key -> sid -> effect -> count
        self.min_size = {}                    # sid -> size at which a retract was refused
        self.max_size = {}                    # sid -> size at which an extend was refused
        self.ctx = {}                         # (key, selected sid) -> KeyModel: the key's laws under that selection
        self.slides = set()                   # (key, sid): the key's translate goes ON, step after step, until something stops it (ice)
        self.drift = None                     # (dr, dc): A STANDING PUSH — after every press the body also goes this one step, when free (a wind, a current)
        self.drift_votes = defaultdict(set)   # (dr, dc) -> keys whose two readings it reconciles
        # THE OUTCOME CONDITIONED ON WHAT CAME BEFORE (Tushar, 2026-10-24: "a
        # single input can have 3 different resulting actions — identifying
        # that is a basic acknowledgement; identifying the pattern is
        # intelligence").  Every reading of a key on a thing is kept with its
        # CONDITION: the key pressed before it and the thing's HEADING (the
        # way of its last step).  When a key's readings of a thing differ
        # and no drift or slide explains them, the coarsest condition under
        # which every value of the condition saw one outcome is the key's
        # law — by the press before, by the heading, or by both.  (tw05 L3:
        # the running jump — up after a step forward goes five cells, up
        # after anything else goes three.)
        self.last_key = None
        self.heading = {}                     # sid -> its FACING: the last key that moved or turned it
        self.cond = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))   # key -> sid -> (prev_key, heading) -> effect -> n
        self.arcs = set()                     # (key, sid): the last press's path bent (an arc) — no slide, no flicker, the reading is the body's own
        self.blocked_none = defaultdict(lambda: defaultdict(int))   # key -> sid -> refusals with something in the way

    # THE DRIFT: a key read to move the body two different ways, the two
    # readings differing by one step along one axis, and the step the same
    # under a second key — the body goes the key's own way AND one step of
    # the drift when the drift's way is free.  The law kept is the key's own
    # step (the reading without the drift; the axis-aligned one when both
    # could be); a reading of own, own+drift, the drift alone (own refused)
    # or nothing (both refused) contradicts nothing.  (shw0, the study game
    # drawn blind on 2026-10-24: "identity + drift_e" — every key's law was
    # contradicted by its second press, and no plan was ever made.)
    def _drift_candidates(self, old, e):
        out = []
        d_ = (e[1][0] - old[1][0], e[1][1] - old[1][1])
        if (d_[0] == 0) == (d_[1] == 0):
            return out                        # (no step along one axis: not a drift)
        for dv_, own_ in ((d_, old[1]), ((-d_[0], -d_[1]), e[1])):
            # (own = the reading without the drift; the other reading = own + drift;
            # the drift is ONE STEP of the body's own lattice — as long as the
            # own step: a difference of another length is the tracker's, not a
            # drift's — wa30, the body's nose read into its box)
            if ((own_[0] == 0) != (own_[1] == 0)) and max(abs(dv_[0]), abs(dv_[1])) == max(abs(own_[0]), abs(own_[1])):
                out.append((dv_, own_))
        return out

    def _own_step(self, key, sid, drift):
        """The key's own step on sid under the drift: the axis-aligned
        candidate every translate reading of the key fits as own, own+drift,
        drift alone or nothing."""
        seen_ = [e_[1] for e_, n_ in self.alt[key][sid].items() if e_[0] == "translate" and n_ > 0]
        if not seen_:
            return None
        cands_ = set()
        for v_ in seen_:
            cands_.add(v_); cands_.add((v_[0] - drift[0], v_[1] - drift[1]))
        ok_ = []
        for d_ in cands_:
            if d_ == (0, 0) or ((d_[0] == 0) == (d_[1] == 0)):
                continue
            fits_ = {d_, (d_[0] + drift[0], d_[1] + drift[1]), drift, (0, 0)}
            if all(v_ in fits_ for v_ in seen_):
                ok_.append(d_)
        if not ok_:
            return None
        return min(ok_, key=lambda d: (abs(d[0]) + abs(d[1]), d))

    def _drift_fits(self, key, sid, old, e):
        """Under the known drift, is the new reading no contradiction of the old law?"""
        dr_ = self.drift
        if dr_ is None or old[0] != "translate" or e[0] != "translate":
            return False
        own_ = self._own_step(key, sid, dr_)
        if own_ is None:
            return False
        fits_ = {own_, (own_[0] + dr_[0], own_[1] + dr_[1]), dr_, (0, 0)}
        return e[1] in fits_ and old[1] in fits_

    def _adopt_drift(self, dv_):
        """The drift is read: every key's law on every thing it moves becomes its own step."""
        self.drift = dv_
        self.slides = set()                   # (what looked like a slide was the drift's step)
        for key in list(self.eff):
            for sid_, e_ in list(self.eff[key].items()):
                if e_[0] != "translate":
                    continue
                own_ = self._own_step(key, sid_, dv_)
                if own_ is not None:
                    self.eff[key][sid_] = ("translate", own_)
                    self.support[key][sid_] = max(1, sum(n_ for e2_, n_ in self.alt[key][sid_].items() if e2_[0] == "translate"))
                    self.contradicted[key].discard(sid_)

    # the three conditions, coarsest first: the press before; the facing; both
    _COND_PROJ = (lambda c: (c[0],), lambda c: (c[1],), lambda c: c)

    @staticmethod
    def _is_outcome(e_):
        return e_[0] != "none" and not (e_[0] == "translate" and not (e_[1][0] or e_[1][1]))

    def _cond_recs_all(self, key, sid):
        """[(condition, outcome)] with every sighting counted (for the carry)."""
        recs = self.cond.get(key, {}).get(sid)
        if not recs:
            return []
        return [(c_, e_) for c_, cnt_ in recs.items() for e_, n_ in cnt_.items() if n_ > 0 and self._is_outcome(e_)]

    def _cond_recs(self, key, sid):
        """[(condition, outcome)] — every outcome (a move, a turn, a growth;
        not a refusal, which is the place's) seen under each condition."""
        recs = self.cond.get(key, {}).get(sid)
        if not recs:
            return []
        # (an outcome seen ONCE is no evidence of a condition — a misread
        # length, a turn seen on one press: wa30 L3 was lost to a 7 read
        # once under one pair; an outcome counts when seen twice over)
        tot_ = Counter()
        for cnt_ in recs.values():
            for e_, n_ in cnt_.items():
                if self._is_outcome(e_):
                    tot_[e_] += n_
        return [(c_, e_) for c_, cnt_ in recs.items() for e_, n_ in cnt_.items() if n_ > 0 and self._is_outcome(e_) and tot_[e_] >= 2]

    def _cond_map(self, key, sid):
        """(projection index, {condition value: the one outcome seen under it})
        for the coarsest condition that explains every reading of the key on
        the thing — None when the readings agree (nothing to condition) or no
        condition explains them.  Refusals (no move) are the place's, not
        the condition's, and are left out."""
        pairs_ = self._cond_recs(key, sid)
        outs = {e_ for _, e_ in pairs_}
        if len(outs) < 2:
            return None
        # (readings all the SAME WAY, differing in length, are one motion
        # stopped short — a wall, a slide, a misreading — unless the path was
        # read frame by frame and bent (an arc: the jump's reach); ka59 L1, a
        # part's half-step against a wall "conditioned" on its heading, and the
        # plan built on it died.  A turn beside a move (wa30: a press the body
        # does not face turns it in place; the next moves it) is no such thing)
        if (key, sid) not in self.arcs:
            os_ = list(outs)
            if all(o_[0] == "translate" for o_ in os_) and all(_same_way(os_[0][1], o_[1]) for o_ in os_[1:]):
                return None
        for i_, pr_ in enumerate(self._COND_PROJ):
            m_ = defaultdict(set)
            for c_, e_ in pairs_:
                m_[pr_(c_)].add(e_)
            if len(m_) >= 2 and all(len(v_) == 1 for v_ in m_.values()):
                return i_, {c_: next(iter(v_)) for c_, v_ in m_.items()}
        return None

    def conditioned(self, key, sid):
        return self._cond_map(key, sid) is not None

    def cond_law(self, key, sid, prev_key, heading):
        """The key's law on the thing under the condition: an effect; None
        when the key is not conditioned on the thing; "unknown" when it is
        and nothing seen speaks for this condition.  A condition never seen
        whole is answered by the coarser conditions it shares with the
        record — the press before alone, the facing alone — when every
        reading under that coarser condition agrees (a press there will
        tell otherwise)."""
        cm_ = self._cond_map(key, sid)
        if cm_ is None:
            return None
        i_, m_ = cm_
        c0_ = (prev_key, heading)
        e_ = m_.get(self._COND_PROJ[i_](c0_))
        if e_ is not None:
            return e_
        pairs_ = self._cond_recs(key, sid)
        for pr_ in self._COND_PROJ[:2]:
            seen_ = {e2_ for c2_, e2_ in pairs_ if pr_(c2_) == pr_(c0_)}
            if len(seen_) == 1:
                return next(iter(seen_))
        return "unknown"

    def unrecord(self, sids):
        """This press's readings of these things were not the key's (an
        agent's own step; a thing in an agent's hands): struck from the
        conditioned records as from the laws.  (tw01: a block the helper
        pushed left under our 'right' was read as 'right' conditioned on
        the press before, and the plan carried that condition.)"""
        for (k_, sid_, c_, e_) in getattr(self, "_last_recs", ()):
            if sid_ in sids or (sid_ >= 1000 and sid_ // 1000 in sids):
                try:
                    self.cond[k_][sid_][c_][e_] -= 1
                    if self.cond[k_][sid_][c_][e_] <= 0:
                        del self.cond[k_][sid_][c_][e_]
                except Exception:
                    pass

    def forget(self, sids):
        """These things' motions were not the key's (an agent's own step; a
        thing in an agent's hands): struck from the laws — under every
        selection too (the selection's records kept a helper's step as a
        law of our key, and the plan moved the helper; wa30 L4) — and from
        the conditioned records of this press."""
        sids = set(sids)
        for k_ in list(self.eff):
            for sid_ in [x_ for x_ in self.eff[k_] if x_ in sids or (x_ >= 1000 and x_ // 1000 in sids)]:
                self.eff[k_].pop(sid_, None)
        for c_ in self.ctx.values():
            for k_ in list(c_.eff):
                for sid_ in [x_ for x_ in c_.eff[k_] if x_ in sids or (x_ >= 1000 and x_ // 1000 in sids)]:
                    c_.eff[k_].pop(sid_, None)
        self.unrecord(sids)

    def _laws_under(self, ctx, key, base_, varies_=frozenset()):
        """The laws under one selection's record: what it read, and — for a
        thing whose every press under it was a refusal with something in
        the way — its law over all (nothing was learned of it there)."""
        res_ = dict(ctx.moving(key))
        if _ENV.get("OLD_CTX_RULE"):
            return res_
        for sid, e in base_.items():
            if sid in res_ or sid in varies_ or (sid in ctx.eff.get(key, {}) and ctx.eff[key][sid][0] != "none"):
                continue
            alt_ = ctx.alt.get(key, {}).get(sid, {})
            n_none_ = alt_.get(("none", (0, 0)), 0)
            if sum(alt_.values()) == n_none_ and ctx.blocked_none.get(key, {}).get(sid, 0) >= n_none_:
                res_[sid] = e
        return res_

    def has_cond(self):
        """A conditioned law the plan must carry its condition for: of a thing
        the key still has a law on (an agent's readings under our keys are
        records, not laws — its motion is its own; wa30 L4's three helpers
        put the press-before into every plan state and the approach search
        timed out)."""
        return any(s_ in self.eff.get(k_, {}) and self.conditioned(k_, s_) for k_ in self.cond for s_ in self.cond[k_])

    def observe(self, key, before, after, sel=None, live_bars=(), skip=frozenset(), override=None, blocked=frozenset()):
        # A LAW MAY DEPEND ON WHAT IS SELECTED: the same key moves the
        # selected thing, and what follows it (a mirror image follows the
        # shape when the shape is selected, the axis when the axis is).
        # Records are kept per (key, selected thing) as well as per key;
        # the plan asks with its state's selection
        if sel is not None:
            ctx = self.ctx.get((key, sel))
            if ctx is None:
                ctx = self.ctx[(key, sel)] = KeyModel()
            ctx.observe(key, before, after, live_bars=live_bars, skip=skip, override=override, blocked=blocked)
        self.presses[key] += 1
        b_by = {s.sid: s for s in before}
        # A THING PUMPED DOWN TO NOTHING THAT GROWS AGAIN IS THE SAME THING:
        # a thing born where one of its kind stood when it was last seen
        # (the same columns, the same base) takes that thing's name in the
        # laws — every key's law of it, its floor and ceiling.  (vc33 L3:
        # the water column emptied by one press and refilled by the next
        # was a new thing each time, and no law held it.)
        try:
            fp_ = getattr(self, "footprint", None)
            if fp_ is None:
                fp_ = self.footprint = {}
            a_sids_ = {s.sid for s in after}
            for s in after:
                if s.sid in b_by or s.sid in skip or s.sid >= 1000:
                    continue
                bb_ = s.bbox
                for old_sid, (okind_, obb_) in list(fp_.items()):
                    if old_sid in a_sids_ or okind_ != s.body or old_sid >= 1000:
                        continue
                    same_cols_ = obb_[1] == bb_[1] and obb_[3] == bb_[3]
                    same_rows_ = obb_[0] == bb_[0] and obb_[2] == bb_[2]
                    same_base_ = obb_[2] == bb_[2] or obb_[0] == bb_[0]
                    same_side_ = obb_[3] == bb_[3] or obb_[1] == bb_[1]
                    if (same_cols_ and same_base_) or (same_rows_ and same_side_):
                        for k_ in list(self.eff):
                            if old_sid in self.eff[k_] and s.sid not in self.eff[k_]:
                                self.eff[k_][s.sid] = self.eff[k_][old_sid]
                                self.support[k_][s.sid] = self.support[k_].get(old_sid, 1)
                        if old_sid in self.min_size:
                            self.min_size[s.sid] = self.min_size[old_sid]
                        if old_sid in self.max_size:
                            self.max_size[s.sid] = self.max_size[old_sid]
                        fp_.pop(old_sid, None)
                        break
            for s in after:
                if s.sid < 1000 and s.sid not in skip:
                    fp_[s.sid] = (s.body, s.bbox)
        except Exception:
            pass
        heads_ = {}
        self._last_recs = []                     # (this press's conditioned records, for the fiddle to unsay)
        for s in after:
            if s.sid in skip:
                continue                # (texture: a strewn colour's lone cells have no laws)
            p = b_by.get(s.sid)
            # (a bar is the HUD's — a clock, a strip — unless it is LIVE: a
            # mirror's axis, a selected wall; then it has laws like anything)
            if p is None or "hud" in s.tags or ("bar" in s.tags and s.sid not in live_bars and (s.sid // 1000 if s.sid >= 1000 else s.sid) not in live_bars):
                continue
            e = effect_of(p, s)
            # A THING THAT FELL INTO TWO BY A MOVE LOST NOTHING: the kept piece
            # stands as it stood (no retract — tw01's pair of blocks, one
            # pushed off, read as a retract and the pushed one as an extend,
            # and no plan pushed anything after that); the piece that went is
            # a thing of its own, pushed — the push record's, not a law here
            if _ENV.get("SPLIT_LAW") and getattr(s, "split_lost", None) is not None and e[0] in ("retract", "morph", "shrink"):
                # (EXPERIMENT, off by default — SPLIT_LAW: on tw01 L1 the retract
                # and extend the split is read as are the wrong law that WORKS
                # (37o); without them the plan loops on the pair.  The split
                # press stays open: tw01 L3)
                s.split_lost = None
                continue
            # WHAT MOVED IS THE THING MOVED: when the tracker read the thing
            # as shifted whole by a vector (not turned) and its size is the
            # same, that shift is its law — however it is drawn within (an
            # eye that turned to the other side: a translate, not a turn)
            mv = getattr(s, "moved_by", None)
            if mv is not None and p.size == s.size and e[0] in ("rotate", "translate", "morph") and e != ("translate", mv):
                e = ("translate", (int(mv[0]), int(mv[1])))
            if override and s.sid in override:
                e = override[s.sid]             # (a member's motion is its body's)
            if e[0] == "morph":
                self.contradicted[key].add(s.sid)
                continue
            # EVERYTHING A KEY MOVES MOVES ON ONE LATTICE: a displacement that
            # is no multiple of the step the key's established law takes (the
            # body's 4, and a thing read as gone 7) is a misreading of the
            # tracker — a like thing some way off taken for this one — and no
            # law (wa30 L2: a crate "moved 7" under ACTION2, the carry's plans
            # then built on it).  The step: the smallest nonzero magnitude
            # among this key's translate laws held with support.
            if e[0] == "translate" and (e[1][0] or e[1][1]):
                mags_ = [max(abs(v_[1][0]), abs(v_[1][1])) for sid_, v_ in self.eff[key].items()
                         if v_[0] == "translate" and (v_[1][0] or v_[1][1]) and self.support[key].get(sid_, 0) >= 1 and sid_ != s.sid]
                if mags_:
                    step_ = min(mags_)
                    m_ = max(abs(e[1][0]), abs(e[1][1]))
                    if step_ > 1 and m_ % step_ != 0:
                        continue
                # ...AND ALONG ONE AXIS: a key whose every held translation runs
                # along one axis (a 'right' key) does not send one thing
                # diagonally — a displacement with a component across that axis
                # is the tracker's misnaming of two like things (tw01: 'ACTION3
                # moves the block by (-4, 8)' after two blocks traded names)
                axes_ = [v_[1] for sid_, v_ in self.eff[key].items()
                         if v_[0] == "translate" and (v_[1][0] or v_[1][1]) and self.support[key].get(sid_, 0) >= 1 and sid_ != s.sid]
                if axes_ and not _ENV.get("NO_AXIS_RULE"):
                    if all(v_[0] == 0 for v_ in axes_) and e[1][0] != 0:
                        continue
                    if all(v_[1] == 0 for v_ in axes_) and e[1][1] != 0:
                        continue
            old = self.eff[key].get(s.sid)
            # UNDER A KNOWN DRIFT a reading is the own step plus the drift
            # when the drift's way was free: the reading is reduced to the
            # own step before it is weighed (a seeded own step would else
            # be replaced by own+drift, and the plan add the drift twice)
            dr0_ = getattr(self, "drift", None)
            if dr0_ is not None and e[0] == "translate" and (e[1][0] or e[1][1]):
                red_ = (e[1][0] - dr0_[0], e[1][1] - dr0_[1])
                if red_ != (0, 0) and (red_[0] == 0) != (red_[1] == 0):
                    e = ("translate", red_)
                elif e[1] == dr0_ and old is not None and old[0] == "translate" and old[1] != e[1]:
                    # (the drift alone: the own step was refused there)
                    self.contradicted[key].add(("refused", s.sid))
                    continue
            self.alt[key][s.sid][e] += 1
            if e[0] == "none" and s.sid in blocked:
                self.blocked_none[key][s.sid] += 1      # (a refusal with something in the way)
            # (the reading with its condition: the press before, the FACING —
            # the last key that moved or turned the thing; a refused press
            # turns nothing)
            self.cond[key][s.sid][(self.last_key, self.heading.get(s.sid))][e] += 1
            self._last_recs.append((key, s.sid, (self.last_key, self.heading.get(s.sid)), e))
            if (e[0] == "translate" and (e[1][0] or e[1][1])) or e[0] == "rotate":
                heads_[s.sid] = key
            if _ENV.get("SLIDE_DEBUG") and old is not None and old != e:
                print(f"   [law] {key} S{s.sid}: old {old} new {e} mv {mv} slides {(key, s.sid) in self.slides}")
            # A TURN IN PLACE IS NO MOVE: a key seen to turn a thing where it
            # stands (its box's centre within its own extent of where it was)
            # and, another time, to move it, MOVES it — the turn is the press
            # that faces the thing (a refusal that changes the picture), not a
            # second law.  (wa30 L4: the body's carried 'left' was replaced by
            # the turn its first press showed, the move that followed then
            # "contradicted" the turn, and the plan could never go left.)
            if not _ENV.get("NO_TURN_RULE") and old is not None and {old[0], e[0]} == {"translate", "rotate"}:
                in_place_ = False
                if e[0] == "rotate":
                    pb_, sb_ = p.bbox, s.bbox
                    ext_ = max(pb_[2] - pb_[0], pb_[3] - pb_[1]) + 1
                    in_place_ = abs((pb_[0] + pb_[2]) - (sb_[0] + sb_[2])) / 2.0 <= ext_ and abs((pb_[1] + pb_[3]) - (sb_[1] + sb_[3])) / 2.0 <= ext_
                else:
                    in_place_ = self.alt[key][s.sid].get(old, 0) >= 1      # (the turn recorded before was a turn of this thing)
                if in_place_:
                    if e[0] == "translate" and (e[1][0] or e[1][1]):
                        self.eff[key][s.sid] = e
                        self.contradicted[key].discard(s.sid)
                    # (the thing answered the key — it turned to it: the law
                    # is no provisional record any more, whatever refusals follow)
                    self.support[key][s.sid] = max(1, self.support[key][s.sid])
                    self.contradicted[key].add(("refused", s.sid))
                    continue
            # A SLIDE: the same key moved the thing the same way but not the
            # same distance — it goes on until something stops it.  The law
            # is the step (the shortest move seen), repeated by the plan
            # while the way is clear; a move of another length is no
            # contradiction of it
            # (a press whose path BENT — an arc — is no slide: the frames showed
            # the body's own way, step by step, and it was not one straight run)
            if old is not None and old[0] == "translate" and e[0] == "translate" and old != e and (key, s.sid) not in self.slides and (key, s.sid) not in self.arcs:
                if _same_way(old[1], e[1]):
                    ext_ = (s.bbox[2] - s.bbox[0] + 1) if old[1][0] else (s.bbox[3] - s.bbox[1] + 1)
                    u = _unit_step(old[1], e[1], ext_)
                    # (both moves are whole steps and one is several: a move
                    # short of a step is a blocked press, not a slide)
                    g_ = abs(u[0]) + abs(u[1])
                    la_, lb_ = abs(old[1][0]) + abs(old[1][1]), abs(e[1][0]) + abs(e[1][1])
                    if g_ >= 2 and la_ % g_ == 0 and lb_ % g_ == 0 and max(la_, lb_) >= 2 * g_:
                        self.eff[key][s.sid] = ("translate", u)
                        self.support[key][s.sid] += 1
                        self.slides.add((key, s.sid))
                        continue
            if (key, s.sid) in self.slides and e[0] == "translate" and old is not None and old[0] == "translate" and _same_way(old[1], e[1]):
                self.support[key][s.sid] += 1
                continue
            if old is None:
                self.eff[key][s.sid] = e
                self.support[key][s.sid] = 1
            elif old == e:
                self.support[key][s.sid] += 1
            elif self.support[key][s.sid] <= 0 and e[0] != "none":
                # a provisional record (carried from another level): the
                # first press that does something else replaces it (a
                # refusal is a refusal of the carried law, as of any)
                self.eff[key][s.sid] = e
                self.support[key][s.sid] = 1
            elif old[0] != "none" and e[0] != "none" and (key, s.sid) not in self.arcs and (self.alt[key][s.sid][old] >= 3 * self.alt[key][s.sid][e]
                                                          or (old[0] == "translate" and e[0] == "translate" and self.alt[key][s.sid][old] >= 2
                                                              and max(abs(e[1][0]), abs(e[1][1])) > 1.5 * max(abs(old[1][0]), abs(old[1][1])))):
                # a well-supported effect is not undone by one odd reading
                # (an identity flicker: a step far longer than the key's
                # step is two things confused, not a new law — unless the
                # path was read frame by frame: an arc's reading is the body's own)
                continue
            elif old[0] == "none":
                # a key that did nothing once and something later: it is
                # conditional; keep the effect, remember the refusal
                self.eff[key][s.sid] = e
                self.support[key][s.sid] = 1
                self.contradicted[key].add(("refused", s.sid))
            elif e[0] == "none":
                self.contradicted[key].add(("refused", s.sid))
                # a PROVISIONAL law (carried from another level, never seen
                # to hold here) refused twice is no law here: dropped, so
                # the plan stops pressing it and looks for another way —
                # refused with NOTHING IN THE WAY: a press against a thing
                # says nothing of the law (wa30 L4: the body's 'left', held
                # over three levels, dropped by two presses against blocks
                # in a crowded yard, and no plan could go left)
                if self.support[key][s.sid] <= 0 and self.alt[key][s.sid][e] >= 2 and s.sid not in blocked:
                    del self.eff[key][s.sid]
                    self.contradicted[key].discard(("refused", s.sid))
                    continue
                # a retract refused at this size: the thing has a minimum;
                # an extend refused with nothing in the way: a maximum
                if old[0] == "retract":
                    self.min_size[s.sid] = max(self.min_size.get(s.sid, 0), s.size)
                elif old[0] == "extend":
                    self.max_size[s.sid] = min(self.max_size.get(s.sid, 10**9), s.size)
            elif old[0] == "translate" and e[0] == "translate" and self._drift_fits(key, s.sid, old, e):
                # (the drift explains the difference: the law stands as the own step)
                own_ = self._own_step(key, s.sid, self.drift)
                if own_ is not None and self.eff[key][s.sid] != ("translate", own_):
                    self.eff[key][s.sid] = ("translate", own_)
                self.support[key][s.sid] += 1
            else:
                if old[0] == "translate" and e[0] == "translate" and self.drift is None and not _ENV.get("NO_DRIFT"):
                    for dv_, own_ in self._drift_candidates(old, e):
                        self.drift_votes[dv_].add(key)
                    # (a drift is read when two keys agree on it; the one with
                    # the most keys, then the one whose own steps are shortest)
                    best_ = [dv_ for dv_, ks_ in self.drift_votes.items() if len(ks_) >= 2]
                    if best_:
                        def _score(dv_):
                            tot_ = 0
                            for k2_ in self.drift_votes[dv_]:
                                o2_ = self._own_step(k2_, s.sid, dv_)
                                tot_ += (abs(o2_[0]) + abs(o2_[1])) if o2_ else 99
                            return (-len(self.drift_votes[dv_]), tot_)
                        dv_ = min(best_, key=_score)
                        # (every voting key's readings of this thing must fit the drift)
                        if all(self._own_step(k2_, s.sid, dv_) is not None for k2_ in self.drift_votes[dv_] | {key}):
                            self._adopt_drift(dv_)
                            continue
                if old[0] == "translate" and e[0] == "translate" and not _ENV.get("NO_COND") and self.conditioned(key, s.sid):
                    # (the condition explains the two readings: the law stands,
                    # the plan asks cond_law with its state's press-before and heading)
                    if _ENV.get("COND_DEBUG"):
                        print(f"   [cond] {key} S{s.sid}: old {old} new {e} under {(self.last_key, self.heading.get(s.sid))}; map {self._cond_map(key, s.sid)}; arcs {(key, s.sid) in self.arcs}")
                    self.support[key][s.sid] += 1
                    self.contradicted[key].discard(s.sid)
                    continue
                self.contradicted[key].add(s.sid)
        # WHAT ONE GAINS ANOTHER LOSES: a key under which a thing extended by
        # n cells while a thing of the same kind went from n cells to nothing
        # has pumped the one into the other — the vanished thing's law under
        # the key is a retract (to nothing; the plan then knows the pump is
        # spent when it stands empty).  vc33 L3: the column pumped away on
        # the fiddle's press was never seen to retract, and the plan pressed
        # the key for ever.
        try:
            a_by_ = {s.sid: s for s in after}
            gained_ = [(s.sid, s.body, s.size - b_by[s.sid].size, self.eff[key].get(s.sid))
                       for s in after if s.sid in b_by and self.eff[key].get(s.sid, ("none",))[0] == "extend" and s.size > b_by[s.sid].size]
            for p in before:
                if p.sid in a_by_ or p.sid in skip or "hud" in p.tags or p.size < 2:
                    continue
                for gsid_, gk_, gn_, ge_ in gained_:
                    if gk_ == p.body and abs(gn_ - p.size) <= max(2, 0.25 * p.size) and p.sid not in self.eff[key]:
                        self.eff[key][p.sid] = ("retract", (ge_[1][0], ge_[1][1]))
                        self.support[key][p.sid] = 1
                        self.min_size[p.sid] = 0
                        break
            # ...and a thing BORN with n cells while one of its kind retracted
            # by n was pumped into being: its law under the key is an extend
            lost_ = [(s.sid, s.body, b_by[s.sid].size - s.size, self.eff[key].get(s.sid))
                     for s in after if s.sid in b_by and self.eff[key].get(s.sid, ("none",))[0] == "retract" and s.size < b_by[s.sid].size]
            for s in after:
                if s.sid in b_by or s.sid in skip or "hud" in s.tags or s.size < 2 or s.sid in self.eff[key]:
                    continue
                if _ENV.get("SPLIT_LAW") and getattr(s, "split_from", None) is not None:
                    continue                      # (born by a split: pushed off, not pumped into being — see SPLIT_LAW above)
                for lsid_, lk_, ln_, le_ in lost_:
                    if lk_ == s.body and abs(ln_ - s.size) <= max(2, 0.25 * s.size):
                        self.eff[key][s.sid] = ("extend", (le_[1][0], le_[1][1]))
                        self.support[key][s.sid] = 1
                        break
        except Exception:
            pass
        # (the condition for the next press: this key, and the headings this press set)
        self.last_key = key
        self.heading.update(heads_)
        # legal overlaps in the observed world
        cells_by = [(s.sid, set(s.cells)) for s in after]
        for i in range(len(cells_by)):
            for j in range(i + 1, len(cells_by)):
                if cells_by[i][1] & cells_by[j][1]:
                    self.overlap_ok.add(frozenset((cells_by[i][0], cells_by[j][0])))

    def moving(self, key, sel=None):
        """Effects that change something, for this key — a contradicted
        effect (the same key did something else to the thing) is not
        planned with; a refusal is (the key is conditional).  With `sel`,
        what the key did under that selection, when it was ever pressed
        under it; else its laws over every selection."""
        if sel is not None:
            ctx = self.ctx.get((key, sel))
            bad = self.contradicted.get(key, ())
            base_ = {sid: e for sid, e in self.eff[key].items() if e[0] != "none" and sid not in bad}
            # (a thing whose law under the key DIFFERS between the selections
            # seen — the selected thing itself, what follows it — is the
            # selection's; a thing whose law is the same under every selection
            # seen walks the same whatever is held)
            # (a refusal with something in the way under a selection teaches
            # nothing of the thing there — wa30 L4: 'right' pressed against a
            # block while holding block 5, and the body could not go right in
            # any plan while holding it, nor while holding anything else, the
            # law "varying" by that refusal)
            varies_ = set()
            n_other_ = 0
            for (k_, s_), c_ in self.ctx.items():
                if k_ == key and s_ != sel and c_.presses.get(key, 0) > 0:
                    n_other_ += 1
                    other_ = self._laws_under(c_, key, base_)
                    varies_ |= {sid for sid in set(other_) | set(base_) if other_.get(sid) != base_.get(sid)}
            # ("the same under every selection seen" wants two selections seen at
            # least; under one, what differs makes the whole key unknown here, as
            # before — re86 L1 planned two presses more on one selection's word)
            if varies_ and n_other_ < 2 and (ctx is None or ctx.presses.get(key, 0) == 0):
                return {}
            if _ENV.get("CTX_DEBUG"):
                print(f"      [ctx] {key} under {sel}: ctx {'yes' if ctx is not None else 'no'} presses {ctx.presses.get(key, 0) if ctx is not None else 0} base {base_} varies {sorted(varies_)} ctx_laws {dict(ctx.moving(key)) if ctx is not None else None} alt {dict(ctx.alt.get(key, {})) if ctx is not None else None}"[:500])
            if ctx is not None and ctx.presses.get(key, 0) > 0:
                return self._laws_under(ctx, key, base_, varies_)
            # never pressed under THIS selection: what varies is unknown here
            # (a press under this selection will tell, not guessed); the rest
            # keeps its law (wa30 L4: under a block never held before, every
            # key was unknown and the body could not move in the plan)
            if varies_ and not _ENV.get("OLD_CTX_RULE"):
                return {sid: e for sid, e in base_.items() if sid not in varies_}
        bad = self.contradicted.get(key, ())
        return {sid: e for sid, e in self.eff[key].items() if e[0] != "none" and sid not in bad}

    def keys(self):
        return [k for k in self.eff if self.moving(k)]


def plan(model, sprites, goal_test, frame=(64, 64), max_states=400000, max_depth=90, solids=None, heuristic=None):
    """Best-first search over key sequences from the current sprites
    (breadth-first when no heuristic is given).  `goal_test(cells)` takes
    {sid: cells}; `heuristic(state)` a distance-to-goal estimate.
    `solids`: sids that refuse overlap.  Returns the key list or None."""
    import heapq, itertools
    keys = model.keys()
    if not keys:
        return None
    start = {s.sid: frozenset(s.cells) for s in sprites}
    if solids is None:
        solids = {s.sid for s in sprites}
    H, W = frame

    def sig(state):
        return tuple(sorted((sid, hash(cs)) for sid, cs in state.items()))
    seen = {sig(start)}
    tick = itertools.count()
    h0 = heuristic(start) if heuristic else 0
    q = [(h0, next(tick), start, [])]
    n = 0
    best_h = h0
    stats = {"refused_bounds": 0, "refused_limit": 0, "refused_overlap": 0}
    plan.last_stats = stats
    while q:
        hcur, _, state, path = heapq.heappop(q)
        if heuristic:
            hh = heuristic(state)
            if hh < best_h:
                best_h = hh; stats["best_h"] = hh; stats["best_path"] = list(path)
        if goal_test(state):
            return path
        if len(path) >= max_depth:
            continue
        for k in keys:
            eff = model.moving(k)
            new = dict(state)
            ok = True
            changed = []
            for sid, e in eff.items():
                if sid not in new:
                    continue
                if e[0] == "retract" and len(new[sid]) <= model.min_size.get(sid, 0):
                    ok = False; stats["refused_limit"] += 1; break
                if e[0] == "extend" and len(new[sid]) >= model.max_size.get(sid, 10**9):
                    ok = False; stats["refused_limit"] += 1; break
                cs = apply(new[sid], e)
                if any(r < 0 or c < 0 or r >= H or c >= W for r, c in cs):
                    ok = False; stats["refused_bounds"] += 1; break
                new[sid] = frozenset(cs)
                changed.append(sid)
            if not ok or not changed:
                continue
            # collisions: a changed solid overlapping any other solid
            for sid in changed:
                if sid not in solids:
                    continue
                cs = new[sid]
                for o, ocs in new.items():
                    if o == sid or o not in solids or frozenset((sid, o)) in model.overlap_ok:
                        continue
                    if o in eff and o in changed:
                        continue        # moved together: one body
                    if cs & ocs:
                        ok = False; stats["refused_overlap"] += 1; stats.setdefault("overlap_pairs", Counter())[(sid, o)] += 1; break
                if not ok:
                    break
            if not ok:
                continue
            s2 = sig(new)
            if s2 in seen:
                continue
            seen.add(s2)
            n += 1
            if n > max_states:
                stats["states"] = n
                return None
            f = (len(path) + 1) + (heuristic(new) if heuristic else 0)
            heapq.heappush(q, (f, next(tick), new, path + [k]))
    stats["states"] = n
    return None


time_left = None        # set by the game loop: a callable giving the level's thinking time left (s)


def plan_general(start, successors, goal_test, heuristic=None, max_states=200000, max_depth=80, max_seconds=90.0):
    """Best-first search over an arbitrary successor function.
    `successors(state) -> [(action, new_state)]`; states must be hashable
    via `sig(state)` = tuple(sorted((k, hash(v)) ...)).
    NO SEARCH TAKES THE LEVEL'S WHOLE CLOCK (bp35 L2: a walk searched 16k
    states twice and the readings never got their turn — "lost in 1
    action"): a search gets at most a quarter of what is left."""
    import heapq, itertools, time
    if time_left is not None:
        try:
            max_seconds = min(max_seconds, max(3.0, 0.25 * float(time_left())))
        except Exception:
            pass
    t_end = time.time() + max_seconds

    def sig(state):
        # (the state as one hash: its items are frozensets of cells and
        # small values — no sorting, no strings; the old way spent a fifth
        # of a search on signatures)
        if _ENV.get("OLD_SIG"):
            return tuple(sorted((str(k), hash(v)) for k, v in state.items()))
        try:
            return hash(frozenset(state.items()))
        except TypeError:
            return tuple(sorted((str(k), hash(v)) for k, v in state.items()))
    seen = {sig(start)}
    tick = itertools.count()
    q = [((heuristic(start) if heuristic else 0), next(tick), start, [])]
    n = 0
    best = (heuristic(start) if heuristic else 0, [])
    while q:
        _, _, state, path = heapq.heappop(q)
        if goal_test(state):
            plan_general.last = {"states": n, "best": best}
            return path
        if (n & 255) == 0 and time.time() > t_end:
            plan_general.last = {"states": n, "best": best, "timed_out": True}
            return None                 # (a search is bounded in time as in states)
        if len(path) >= max_depth:
            continue
        for act, new in successors(state):
            s2 = sig(new)
            if _ENV.get("PLAN_TRACE"):
                print(f"         [trace] n={n} depth={len(path)} {act} seen={s2 in seen} moved={[ (k_, _bbox(v_)[:2]) for k_, v_ in new.items() if not isinstance(k_, str) and v_ != state.get(k_)][:3]}")
            if s2 in seen:
                continue
            seen.add(s2)
            n += 1
            if n > max_states:
                plan_general.last = {"states": n, "best": best}
                return None
            h = heuristic(new) if heuristic else 0
            if h < best[0]:
                best = (h, path + [act])
            heapq.heappush(q, (len(path) + 1 + h, next(tick), new, path + [act]))
    plan_general.last = {"states": n, "best": best}
    return None


from .perthread import PerThread as _PerThread
plan_general = _PerThread(plan_general)
