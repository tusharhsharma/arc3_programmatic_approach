"""THE RELATION LANGUAGE — readings generated, not listed.

A first frame is a set of THINGS (clusters of one colour, constellations
of like dots, marks riding a host, pictures) and the RELATIONS that hold
between them: same colour, the same shape up to a turn or a flip, a
cavity that fits, a box that holds, the same pattern of marks, two
pictures of one size, a twin in the HUD, the one thing of its colour.
A READING is a goal predicate over an end state — a inside b, a over b,
a's marks on b's marks, picture a equal to picture b, the mover at the
odd thing — proposed for EVERY pair the relations support and scored by
the evidence: each relation carries a weight (a prior, noted below,
adjusted by what the level teaches: a reading pursued to its end with
nothing happening lowers the weight of the relations that proposed it;
the reading that won raises them, and the weights carry to the next
level).  The old hand list (socket, twin, fit, marks, constellation) is
a set of special cases of this language and every one of its readings
is still produced here, with the same words, so the rest of the core
reads them as before.

Nothing here knows a game.  The weights say how much a human trusts each
cue before touching a key; the feedback says how much this game does.
"""
import os
_ENV = dict(os.environ)      # (the switches are read once: a lookup in os.environ is a Mapping call, and one stood in the planner's successor loop — 20 million calls, 60 s of a 300 s level on bp35 L2)
from collections import Counter, defaultdict

from ..goals import (GoalCluster, GoalHypothesis, goal_clusters, _gdihedral, _gbbox_gap,
                     _gdist, goal_same_kind)


# THE PRIOR: how much each cue says "this goes there" before any press.
# (hand-chosen; the values keep the hand list's ranking: socket and twin
# readings first, then fits and marks, then same-colour pairs, then reach)
PRIOR = {
    "fits_cavity": 3.0,          # a's dims are b's cavity's dims
    "same_colour": 1.0,          # a and b share a colour
    "own_constellation": 2.5,    # b is a loose drawing of a's colour
    "cavity": 1.5,               # b has a cavity (an outline, a ring)
    "same_shape": 1.2,           # the same shape up to a turn or a flip, another colour
    "same_marks": 2.0,           # like marks on a and on b, the same pattern
    "hud_twin": 3.2,             # a's twin sits in the HUD and b is the one ring
    "picture_pair": 2.5,         # two pictures of one size, one of them many-coloured
    "unique": 0.8,               # b is the only thing of its colour
    "smaller": 0.5,              # a is not bigger than b
    "turned": -0.2,              # the fit needs a turn
}


def _marks_fit(a, b):
    """Whether a's marks (their centres) fit onto some of b's marks under
    one of the eight turns/flips: None when they do not; else True when
    the fit needs a turn, False when it lies as it is."""
    ma = getattr(a, "marks", None); mb = getattr(b, "marks", None)
    if not ma or not mb or len(ma) < 1:
        return None
    # (a pair of marks fixes a placement: a piece docks by two of its marks
    # on two of the other's — the rest of either may stay free)
    need = min(len(ma), 2)
    def cen(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return (sum(rs) / float(len(rs)), sum(cs) / float(len(cs)))
    A = [cen(m) for m in ma]; B = [cen(m) for m in mb]
    Bset = {(round(r * 2), round(c * 2)) for r, c in B}
    transforms = [lambda r, c: (r, c), lambda r, c: (c, -r), lambda r, c: (-r, -c), lambda r, c: (-c, r),
                  lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (c, r), lambda r, c: (-c, -r)]
    for ti, T in enumerate(transforms):
        TA = [T(r, c) for r, c in A]
        for r0, c0 in TA:
            for br, bc in B:
                dr, dc = br - r0, bc - c0
                if sum((round((r + dr) * 2), round((c + dc) * 2)) in Bset for r, c in TA) >= need:
                    return ti != 0
    return None


class Thing:
    """A thing on the first frame, as the relations see it."""
    __slots__ = ("cl", "kind", "host", "hud", "border", "terrain", "unique", "cavity", "cav_dims", "cav_centre", "id", "colours", "derived", "marks")

    def __init__(self, cl, kind="cluster"):
        self.cl = cl; self.kind = kind; self.host = None
        self.hud = False; self.border = False; self.terrain = False; self.unique = False; self.derived = False
        self.cavity = cl.enclosed() if kind != "marks" else frozenset()
        if self.cavity:
            rs = [r for r, _ in self.cavity]; cs = [c for _, c in self.cavity]
            self.cav_dims = (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
            self.cav_centre = ((min(rs) + max(rs)) / 2.0, (min(cs) + max(cs)) / 2.0)
        else:
            self.cav_dims = None; self.cav_centre = None
        self.id = id(cl)
        self.colours = (cl.color,)
        self.marks = None

    # the cluster's face
    @property
    def color(self):
        return self.cl.color

    @property
    def cells(self):
        return self.cl.cells

    @property
    def size(self):
        return self.cl.size

    @property
    def dims(self):
        return self.cl.dims

    @property
    def centre(self):
        return self.cl.centre

    @property
    def norm(self):
        return self.cl.norm

    @property
    def bbox(self):
        return self.cl.bbox

    @property
    def n_parts(self):
        return self.cl.n_parts

    def __repr__(self):
        return f"{self.kind}:{self.cl!r}"


def things_of(objs, frame_h, frame_w, playfield=None, terrain=(), hud_cells=frozenset()):
    """The things: constellations of tiny like components (markers, never
    movers), clusters of one colour, and MARK GROUPS (small things of one
    colour riding a larger host of another)."""
    if hud_cells:
        objs = [o for o in objs if not (set(o.cells) <= hud_cells)]

    def in_hud_b(bbox):
        if playfield is None:
            return bbox[0] < 8
        r0, r1, c0, c1 = playfield
        return bbox[2] < r0 or bbox[0] >= r1 or bbox[3] < c0 or bbox[1] >= c1
    tiny = [o for o in objs if o.size <= 6 and not in_hud_b(o.bbox)]
    const = [k for k in goal_clusters(tiny, gap=12) if k.n_parts >= 3]
    tight = [k for k in goal_clusters(tiny, gap=2) if k.n_parts >= 3 and k.enclosed()]
    const_cells = set()
    for k in const:
        const_cells |= k.cells
    pool = [o for o in objs if not (o.cells <= const_cells)]
    cl = [c for c in goal_clusters(pool)
          if not (c.dims[1] >= frame_w - 2 and c.dims[0] <= 3)
          and not (c.dims[0] >= frame_h - 2 and c.dims[1] <= 3)
          and c.size <= 600]
    things = []
    for k in const:
        things.append(Thing(k, "constellation"))
    for t in tight:
        if not any(t.cells == k.cells for k in const):
            things.append(Thing(t, "constellation"))
    for c in cl:
        things.append(Thing(c, "cluster"))
    by_col = Counter(t.color for t in things if t.kind == "cluster")
    # (a DERIVED thing — an image a law puts beyond the frame — is neither
    # HUD nor border for lying outside the playfield)
    derived_cells = set()
    for o in objs:
        if getattr(o, "derived", False):
            derived_cells |= set(o.cells)
    for t in things:
        t.derived = bool(derived_cells) and bool(set(t.cells) & derived_cells)
        t.hud = in_hud_b(t.bbox) and not t.derived
        t.border = (t.bbox[0] <= 0 or t.bbox[1] <= 0 or t.bbox[2] >= frame_h - 1 or t.bbox[3] >= frame_w - 1) and not t.derived
        t.terrain = t.color in terrain
        t.unique = t.kind == "cluster" and by_col[t.color] == 1
    # mark groups: small clusters of one colour touching a larger host of
    # another colour, grouped by host
    clusters = [t for t in things if t.kind == "cluster"]

    def host_of(mk):
        best = None
        for h in clusters:
            if h is mk or h.color == mk.color or h.size <= mk.size:
                continue
            if _gbbox_gap(h.bbox, mk.bbox) <= 1 and any(abs(r - hr) + abs(c - hc) == 1 for r, c in mk.cells for hr, hc in h.cells):
                if best is None or h.size < best.size:
                    best = h
        return best
    groups = defaultdict(list)
    for mk in clusters:
        if mk.size <= 16 and mk.n_parts == 1 and not mk.hud:
            h = host_of(mk)
            if h is not None:
                groups[(h.id, mk.color)].append((h, mk))
    for (hid, col), hm in groups.items():
        if len(hm) < 2 or any(m.norm != hm[0][1].norm for _, m in hm):
            continue
        cells = set()
        for _, m in hm:
            cells |= m.cells
        t = Thing(GoalCluster(col, cells, len(hm)), "marks")
        t.host = hm[0][0]
        t.marks = [frozenset(m.cells) for _, m in hm]      # (each mark on its own: a piece docks where SOME of the other's marks are)
        things.append(t)
    # PICTURES: touching clusters of several colours whose union fills a box
    # (a reference drawing); a filled one-colour box is a canvas candidate
    filled_singles = [t for t in clusters if t.size == t.dims[0] * t.dims[1] and t.size >= 16 and not t.hud]
    n = len(clusters)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, b = clusters[i], clusters[j]
            if a.hud or b.hud or a.size > 400 or b.size > 400:
                continue
            if _gbbox_gap(a.bbox, b.bbox) == 0 and any(abs(r - r2) + abs(c - c2) == 1 for r, c in a.cells for r2, c2 in b.cells):
                parent[find(i)] = find(j)
    unions = defaultdict(list)
    for i in range(n):
        unions[find(i)].append(clusters[i])
    for grp in unions.values():
        if len(grp) < 2:
            continue
        cells = set()
        for t in grp:
            cells |= t.cells
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        if len(cells) == h * w and len(cells) >= 16:
            pic = Thing(GoalCluster(grp[0].color, cells, len(grp)), "picture")
            pic.hud = any(t.hud for t in grp)
            pic.colours = tuple(sorted({t.color for t in grp}))
            things.append(pic)
    return things


def relations_of(things):
    """Every relation that holds, as {name: {(a, b)}} over ordered pairs
    (a the would-be mover, b the would-be target) — and the unary ones
    as {(a, None)}."""
    R = defaultdict(set)
    clusters = [t for t in things if t.kind == "cluster"]
    # how many things share each thing's shape (up to a turn or a flip)
    congruent_n = {}
    for a in clusters:
        if a.n_parts != 1 or a.size < 4:
            congruent_n[a.id] = 0; continue
        dih = _gdihedral(a.norm)
        congruent_n[a.id] = sum(1 for b in clusters if b.n_parts == 1 and b.size == a.size and b.norm in dih)
    marks = [t for t in things if t.kind == "marks"]
    consts = [t for t in things if t.kind == "constellation"]
    for a in clusters:
        if a.hud or a.border or a.terrain:
            continue
        for b in things:
            if b is a or b.kind == "marks":
                continue
            if b.kind == "cluster":
                if a.color == b.color and not goal_same_kind(a.cl, b.cl):
                    R["same_colour"].add((a, b))
                if b.cav_dims is not None and b.cav_dims == a.dims and len(b.cavity) >= 1 and a.color != b.color:
                    if len(a.cavity) <= 4:
                        R["fits_cavity"].add((a, b))
                if b.cav_dims is not None and len(b.cavity) >= 4:
                    R["cavity"].add((a, b))
                # (a DERIVED thing — an image a law puts somewhere — is made
                # to be fitted: its exact shape, as it is, onto a thing of
                # that shape, however many of them there are)
                derived_fit = getattr(a, "derived", False) and not getattr(b, "derived", False) and a.color != b.color \
                    and a.n_parts == 1 and b.n_parts == 1 and a.size == b.size and a.size >= 4 and b.norm == a.norm and not b.terrain
                if derived_fit or (a.color != b.color and a.n_parts == 1 and b.n_parts == 1 and a.size == b.size and a.size >= 4 \
                        and b.dims in (a.dims, a.dims[::-1]) and b.norm in _gdihedral(a.norm) and not b.hud and not b.border \
                        and congruent_n[a.id] == 2 and _gbbox_gap(a.bbox, b.bbox) > 0):
                    # (the same shape means something only when the two are
                    # the ONLY two of that shape: a board of like tiles says nothing)
                    R["same_shape"].add((a, b))
                    if b.norm != a.norm:
                        R["turned"].add((a, b))
                if b.unique and not b.hud and b.size <= 4 * max(a.size, 16):   # (an odd thing a mover could plausibly reach or cover: no larger than four of it — hand-chosen; the pen of 64 cells against a 16-cell block)
                    R["unique"].add((a, b))
                if a.size <= b.size:
                    R["smaller"].add((a, b))
            elif b.kind == "constellation":
                if a.color == b.color and _gdist(a.centre, b.centre) >= 1:
                    R["own_constellation"].add((a, b))
                    if b.cav_dims == a.dims:
                        R["fits_cavity"].add((a, b))
                if a.color != b.color and a.n_parts == 1 and a.size >= 3 and b.norm == a.norm:
                    R["same_shape"].add((a, b))
    for a in marks:
        for b in marks:
            if a is b or a.host is b.host or a.color != b.color or a.n_parts > b.n_parts:
                continue
            # MARKS ONTO LIKE MARKS, AS A PART OF THEM: a's marks fit onto a
            # subset of b's (a piece with two marks docks where two of the
            # anchor's four are), up to a turn; the same pattern exactly
            # when the counts are equal
            fit_ = _marks_fit(a, b)
            if fit_ is not None:
                R["same_marks"].add((a, b))
                if fit_:
                    R["turned"].add((a, b))
    # a twin in the HUD names the mover; an enclosing thing of a unique
    # colour is where it goes
    by_col = defaultdict(list)
    for t in clusters:
        by_col[t.color].append(t)
    twins_ = []
    for col, cs in by_col.items():
        # (one thing of the colour on the field, and its likeness in the HUD —
        # the HUD may hold other things of that colour: a palette's swatch)
        field_ = [t for t in cs if not t.hud]
        if len(field_) != 1 or not any(t.hud and t.norm == field_[0].norm for t in cs):
            continue
        twins_.append((col, field_[0]))
    for col, mover in twins_:
        for rg in clusters:
            # (the place: the one thing of a colour; or, with as many such things
            # as there are twins, one each — su15 L3's two discs for its two;
            # a place another twin already lies on is taken)
            if rg.color == col or rg.hud or len(by_col[rg.color]) not in (1, len(twins_)) or rg.color in {c_ for c_, _ in twins_}:
                continue
            if any(o_ is not mover and o_.bbox[0] >= rg.bbox[0] and o_.bbox[1] >= rg.bbox[1] and o_.bbox[2] <= rg.bbox[2] and o_.bbox[3] <= rg.bbox[3]
                   for _, o_ in twins_):
                continue
            # (where it goes: a ring's hole; or, with no ring on the board, the
            # one thing of a unique colour large enough to lie on — su15 L2's
            # disc, the ring of L1 drawn solid)
            if rg.cavity:
                R["hud_twin"].add((mover, rg))
            elif rg.size >= mover.size and rg.dims[0] >= mover.dims[0] and rg.dims[1] >= mover.dims[1] \
                    and not any(x.cavity and x.color != col and not x.hud and len(by_col[x.color]) == 1 for x in clusters):
                R["hud_twin"].add((mover, rg))
    # pictures: two filled rectangles of one size (or a whole multiple),
    # one of them of several colours — a canvas and its reference
    filled = [t for t in clusters if t.size == t.dims[0] * t.dims[1] and t.size >= 16 and not t.hud]
    pictures = [t for t in things if t.kind == "picture" and not t.hud]
    for a in filled:
        for b in pictures:
            if a.dims == b.dims or (b.dims[0] % a.dims[0] == 0 and b.dims[1] % a.dims[1] == 0 and b.dims[0] // a.dims[0] == b.dims[1] // a.dims[1]):
                R["picture_pair"].add((a, b))
    return R


def propose(things, R, weights=None, frame_h=64, frame_w=64):
    """Readings for every pair the relations support, scored by the
    evidence.  Returns GoalHypothesis objects (with .evidence) in the old
    vocabulary, best first."""
    W = dict(PRIOR)
    if weights:
        W.update(weights)
    pair_ev = defaultdict(set)
    for name, pairs in R.items():
        for pr in pairs:
            if pr[1] is not None:
                pair_ev[pr].add(name)
    hyps = []

    def add(kind, a, target, note, score, t_size, t_cells, ev):
        h = GoalHypothesis(kind, a.cl, target, note, score, t_size)
        h.t_cells = frozenset(t_cells) if t_cells is not None else (frozenset(target) if not isinstance(target, tuple) else frozenset())
        h.evidence = tuple(sorted(ev))
        hyps.append(h)
    # (a fixed order: ties between readings must not depend on memory addresses)
    for (a, b), ev in sorted(pair_ev.items(), key=lambda kv: (kv[0][0].bbox, kv[0][1].bbox, kv[0][0].color, kv[0][1].color)):
        if a.hud or a.border or a.terrain or a.size > 200 or a.n_parts > 4:
            continue
        d = _gdist(a.centre, b.centre)
        if d < 1:
            continue
        base = sum(W.get(e, 0.0) for e in ev) - min(d, 60) / 60.0
        # the reading's kind follows its strongest structural cue; the
        # words are the hand list's, so the rest of the core reads them
        if "same_marks" in ev:
            add("cover", a, b.cells, f"marks onto like marks C{b.color}[{b.size}@{b.bbox}] of C{b.host.color}" + (" turned" if "turned" in ev else "")
                + (f" from C{a.host.color}" if getattr(a, "host", None) is not None else ""),
                6.0 + base, b.size, b.cells, ev)
        elif "hud_twin" in ev:
            add("align", a, b.centre, f"HUD twin -> ring {b.cl!r}", 6.0 + base, b.size, b.cells, ev)
        elif "fits_cavity" in ev and b.kind == "constellation":
            add("align", a, b.cav_centre, f"socket in constellation {b.cl!r}", 6.0 + base, b.size, b.cells, ev)
        elif "own_constellation" in ev:
            if a.size >= 3:
                add("cover", a, b.cells, f"own-colour constellation {b.cl!r}", 6.0 + base, b.size, b.cells, ev)
        elif "fits_cavity" in ev:
            if len(a.cavity) <= 4 and not any(a.cells == t.cavity for t in things):
                add("align", a, b.cav_centre, f"fits frame {b.cl!r}", 5.5 + base, b.size, b.cells, ev)
        elif "same_shape" in ev and b.kind == "constellation":
            add("cover", a, b.cells, f"congruent constellation {b.cl!r}", 6.0 + base, b.size, b.cells, ev)
        elif "same_shape" in ev:
            add("cover", a, b.cells, f"fits the same shape C{b.color}[{b.size}@{b.bbox}]" + (" turned" if "turned" in ev else ""), 5.5 + base, b.size, b.cells, ev)
        elif "same_colour" in ev:
            if b.size > 10 * a.size or b.hud:
                continue
            socket = b.cav_dims == a.dims
            add("align", a, b.centre, f"same-colour marker {b.cl!r}", 7.0 + base + (1.5 if socket else 0) - (1.0 if len(a.cavity) >= 4 else 0)
                - (0.8 if b.size < 3 else 0),          # (a lone dot is a weak marker)
                b.size, b.cells, ev | ({"socket"} if socket else set()))
        elif "picture_pair" in ev:
            # (the paint protocol executes this; the reading names the roles)
            add("equal", a, b.cells, f"picture equal C{b.color}[{b.size}@{b.bbox}] colours {list(b.colours)}", 4.0 + base, b.size, b.cells, ev)
        elif "unique" in ev and len(ev - {"unique", "smaller"}) == 0:
            # the odd thing, by itself: a weak reading (a reach) kept for
            # after the keys have said what moves
            add("align", a, b.centre, f"the odd thing C{b.color}[{b.size}@{b.bbox}]", 3.0 + base, b.size, b.cells, ev)
            # INTO THE ODD THING: a filled thing larger than a in both
            # dimensions is a place a fits INSIDE (a pen, a tray): a's cells
            # within its box — the reading "align" cannot say for a thing
            # smaller than its target on a lattice (tw01: two blocks into a
            # pen twice their size)
            if not _ENV.get("NO_INSIDE") and b.size == b.dims[0] * b.dims[1] and b.size > a.size and b.dims[0] >= a.dims[0] and b.dims[1] >= a.dims[1] \
                    and (b.dims[0] > a.dims[0] or b.dims[1] > a.dims[1]):
                add("inside", a, b.cells, f"into the odd thing C{b.color}[{b.size}@{b.bbox}]", 3.0 + base + 0.1, b.size, b.cells, ev)
    hyps.sort(key=lambda h: -h.prio)
    seen, out, per_mover = set(), [], Counter()
    for h in hyps:
        key = (h.m_color, h.m_start, h.note)
        if key in seen or per_mover[(h.m_color, h.m_start)] >= 3:
            continue
        seen.add(key)
        per_mover[(h.m_color, h.m_start)] += 1
        out.append(h)
    return out[:16]


def read_frame_goals(objs, frame_h, frame_w, playfield=None, terrain=(), hud_cells=frozenset(), weights=None):
    """The relation language's replacement for goal_scan."""
    things = things_of(objs, frame_h, frame_w, playfield, terrain, hud_cells)
    R = relations_of(things)
    return propose(things, R, weights, frame_h, frame_w)


class RelationWeights:
    """What this game has taught about the cues: a reading pursued to its
    end with nothing happening lowers its cues; the reading that won
    raises them.  Carried to the next level."""

    def __init__(self, base=None):
        self.w = dict(base) if base else {}

    def get(self):
        return dict(self.w)

    def failed(self, h):
        for e in getattr(h, "evidence", ()):
            self.w[e] = self.w.get(e, PRIOR.get(e, 0.0)) - 0.5

    def won(self, h):
        for e in getattr(h, "evidence", ()):
            self.w[e] = self.w.get(e, PRIOR.get(e, 0.0)) + 0.5
