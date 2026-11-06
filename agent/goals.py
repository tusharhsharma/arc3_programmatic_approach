"""GOAL INDUCTION FROM FREE INFORMATION (the failure atlas, 2026-09-05).

In 18 of 24 real games winning is a RELATION visible on the first
frame: a movable thing and a marker that share a colour, a frame whose
interior fits a block, a constellation of dots the shape of a piece, a
thing whose twin sits in the HUD and a ring that waits for it.  A human
reads these before touching a key.  This module reads them too — as
HYPOTHESES, ranked and pursued cheapest-first — and pursues one without
knowing the mechanics: an EFFECT LEARNER tries the available actions
(and click points) and records what each does to the target entity; a
greedy descent then picks whatever moves the entity toward the marker.
The level ending confirms the hypothesis; reaching the marker with
nothing happening marks it satisfied and the next candidate (a second
tip, a second cross) takes over; when every candidate is satisfied and
the level stands, the reading was wrong.  Everything is bounded: a
budget per hypothesis, a cap on hypotheses per level.

Entities are CLUSTERS of same-colour components (a diamond drawn as
four dots, an outline drawn as twelve pixels), not raw components.

Effects come in three shapes.  A KEY does the same thing wherever the
mover stands (arrows).  A click ON A THING is a button: the same point
does the same thing (a slider bracket, a selection click).  A click
NEAR THE MOVER is read relative to the mover ('pull it this way'): the
offset is the key, so what was learned at one place transfers to the
next.  A key that moved nothing large may have changed WHICH thing the
others move (a selection): it is kept as a mode candidate and pressed
when the mover will not budge; the old effects are not thrown away but
re-checked, so a merely blocked key costs one action, not a re-survey.

Nothing here knows a game.  It knows: same colour belongs together, a
frame wants what fits it, a twin in the HUD names the thing that
matters, two things of the same kind are not a mover and its marker,
and that the way to find out what an action does is to do it once and
look.
"""
from collections import Counter, defaultdict


class GoalCluster:
    __slots__ = ("color", "cells", "bbox", "n_parts")

    def __init__(self, color, cells, n_parts):
        self.color = color
        self.cells = frozenset(cells)
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        self.bbox = (min(rs), min(cs), max(rs), max(cs))
        self.n_parts = n_parts

    @property
    def size(self):
        return len(self.cells)

    @property
    def dims(self):
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0 + 1, c1 - c0 + 1)

    @property
    def centre(self):
        r0, c0, r1, c1 = self.bbox
        return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

    @property
    def norm(self):
        r0, c0 = self.bbox[0], self.bbox[1]
        return frozenset((r - r0, c - c0) for r, c in self.cells)

    def enclosed(self):
        """Cells inside the bbox that are not the cluster and cannot reach
        the bbox border through non-cluster cells: the cavity of a frame,
        ring or outline (diamond outlines included)."""
        r0, c0, r1, c1 = self.bbox
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            return frozenset()
        free = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if (r, c) not in self.cells}
        border = [p for p in free if p[0] in (r0, r1) or p[1] in (c0, c1)]
        seen = set(border)
        stack = list(border)
        while stack:
            r, c = stack.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in free and q not in seen:
                    seen.add(q)
                    stack.append(q)
        return frozenset(free - seen)

    def __repr__(self):
        return f"C{self.color}[{self.size}@{self.bbox}]"


def _gbbox_gap(b1, b2):
    dr = max(b2[0] - b1[2], b1[0] - b2[2], 0)
    dc = max(b2[1] - b1[3], b1[1] - b2[3], 0)
    return max(dr, dc)


def goal_clusters(objs, gap=2, max_part=400):
    """Same-colour components whose bboxes lie within `gap` of each other
    form one entity."""
    by_col = defaultdict(list)
    for o in objs:
        if o.size <= max_part:
            by_col[o.color].append(o)
    out = []
    for col, os_ in by_col.items():
        parent = list(range(len(os_)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(os_)):
            for j in range(i + 1, len(os_)):
                # (two UNITS of a kind — whole things by the tracker's record —
                # are two things however close: a pair of blocks is a pair)
                if getattr(os_[i], "unit", False) and getattr(os_[j], "unit", False):
                    continue
                if _gbbox_gap(os_[i].bbox, os_[j].bbox) <= gap:
                    parent[find(i)] = find(j)
        groups = defaultdict(list)
        for i, o in enumerate(os_):
            groups[find(i)].append(o)
        for g in groups.values():
            cells = set()
            for o in g:
                cells |= o.cells
            out.append(GoalCluster(col, cells, len(g)))
    return out


def _grebase(cells):
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _gdihedral(norm):
    """The eight images of a shape under rotation and reflection."""
    out = set()
    cur = norm
    for _ in range(4):
        out.add(cur)
        out.add(_grebase([(r, -c) for r, c in cur]))
        cur = _grebase([(c, -r) for r, c in cur])
    return out


def _gbar(c):
    h, w = c.dims
    return c.size == h * w and max(h, w) >= 3 * min(h, w)


def goal_same_kind(a, b):
    """Two things of one kind — congruent up to rotation or mirroring,
    or two solid bars of one thickness (platforms, walls) — are peers
    (the two brackets of a slider, the two rings of a panel), never a
    mover and its marker."""
    if _gbar(a) and _gbar(b) and min(a.dims) == min(b.dims):
        return True
    return a.size == b.size and a.dims in (b.dims, b.dims[::-1]) and b.norm in _gdihedral(a.norm)


class GoalHypothesis:
    def __init__(self, kind, mover, target, note, prio, t_size=0):
        self.kind = kind            # 'align' | 'cover'
        self.m_color = mover.color
        self.m_size = mover.size
        self.m_start = mover.centre
        self.m_cells = mover.cells
        self.target = target        # (r, c) centre or a frozenset of cells
        self.t_size = t_size
        self.note = note
        self.prio = prio
        self.spent = 0
        self.refuted = False
        self.satisfied = False
        self.confirmed = False

    def rule(self):
        """Which first-frame reading produced this hypothesis."""
        return self.note.split(" C")[0]

    def mover_bigger(self):
        return self.m_size > self.t_size

    def target_centre(self):
        if isinstance(self.target, tuple):
            return self.target
        rs = [r for r, _ in self.target]; cs = [c for _, c in self.target]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def __repr__(self):
        return f"{self.kind}(M={self.m_color}/{self.m_size}@{self.m_start} -> {self.note})"


def _gdist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def goal_scan(objs, frame_h, frame_w, playfield=None, terrain=(), hud_cells=frozenset()):
    """First-frame correspondences -> ranked hypotheses.  `terrain`:
    colours the rest of the agent has walked on or bumped — walls and
    floors are not movers (a later level of the same game).  `hud_cells`:
    cells of displays (a strip of dashes, a bar) — never things in play."""
    if hud_cells:
        objs = [o for o in objs if not (set(o.cells) <= hud_cells)]
    def in_hud_b(bbox):
        if playfield is None:
            return bbox[0] < 8
        r0, r1, c0, c1 = playfield
        return bbox[2] < r0 or bbox[0] >= r1 or bbox[3] < c0 or bbox[1] >= c1

    def in_hud(c):
        return in_hud_b(c.bbox)
    # CONSTELLATIONS: loose groups (gap <= 12) of >= 3 tiny COMPONENTS of
    # one colour — corner marks, dot patterns, outlines drawn as dots.
    # They are markers, never movers (their cells leave the entity pool)
    # and their cells are the target.
    tiny = [o for o in objs if o.size <= 6 and not in_hud_b(o.bbox)]
    const = [k for k in goal_clusters(tiny, gap=12) if k.n_parts >= 3]
    const_cells = set()
    for k in const:
        const_cells |= k.cells
    pool = [o for o in objs if not (o.cells <= const_cells)]
    cl = [c for c in goal_clusters(pool)
          if not (c.dims[1] >= frame_w - 2 and c.dims[0] <= 3)
          and not (c.dims[0] >= frame_h - 2 and c.dims[1] <= 3)
          and c.size <= 600]
    by_col = defaultdict(list)
    for c in cl:
        by_col[c.color].append(c)
    enc = {id(c): c.enclosed() for c in cl}
    hyps = []

    def border(c):
        return c.bbox[0] <= 0 or c.bbox[1] <= 0 or c.bbox[2] >= frame_h - 1 or c.bbox[3] >= frame_w - 1

    def mover_ok(m):
        return (not in_hud(m) and not border(m) and m.size <= 200 and m.n_parts <= 4
                and m.color not in terrain)

    def cav_dims(c):
        cav = enc[id(c)]
        if not cav:
            return None
        rs = [r for r, _ in cav]; cs_ = [c_ for _, c_ in cav]
        return (max(rs) - min(rs) + 1, max(cs_) - min(cs_) + 1)

    def add(kind, mover, target, note, prio, t_size=0, t_cells=None):
        h = GoalHypothesis(kind, mover, target, note, prio, t_size)
        h.t_cells = frozenset(t_cells) if t_cells is not None else (frozenset(target) if not isinstance(target, tuple) else frozenset())
        hyps.append(h)

    # (0) MOVER -> CONSTELLATION of its own colour (re86 crosses and dots,
    #     lp85 tile and corner marks, r11l disc and diamond outline).
    #     Two sockets side by side read as one loose constellation with
    #     no cavity: the TIGHT drawings (gap <= 2) inside it are tried
    #     as sockets on their own (s5i5 L3: two diamonds six apart).
    tight = [k for k in goal_clusters(tiny, gap=2) if k.n_parts >= 3 and k.enclosed()]
    seen_k = set()
    for k in const + [t for t in tight if not any(t.cells == c.cells for c in const)]:
        kcav = k.enclosed()
        kdims = None
        if kcav:
            rs = [r for r, _ in kcav]; cs_ = [c for _, c in kcav]
            kdims = (max(rs) - min(rs) + 1, max(cs_) - min(cs_) + 1)
            kctr = ((min(rs) + max(rs)) / 2.0, (min(cs_) + max(cs_)) / 2.0)
        for m in by_col.get(k.color, []):
            if not mover_ok(m):
                continue
            d = _gdist(m.centre, k.centre)
            if d < 1:
                continue
            if kdims == m.dims:
                # the outline is a SOCKET for this mover (s5i5: a dot and
                # its hollow diamond)
                add("align", m, kctr, f"socket in constellation {k!r}", 9.5 - min(d, 60) / 60.0, k.size, k.cells)
            elif m.size >= 3:
                add("cover", m, k.cells, f"own-colour constellation {k!r}", 9.5 - min(d, 60) / 60.0, k.size)

    # (a) SAME COLOUR, DIFFERENT KINDS: every ordered (mover, marker) pair
    #     among 2..6 clusters of a colour; a marker whose cavity fits the
    #     mover is a socket and ranks first; an outline with a cavity
    #     next; both directions are tried because a wrong reading costs
    #     only the effect test.  A marker ten times the mover is a floor,
    #     not a marker.
    for col, cs in by_col.items():
        if not 2 <= len(cs) <= 6:
            continue
        for mover in cs:
            if not mover_ok(mover):
                continue
            for marker in cs:
                if marker is mover or goal_same_kind(marker, mover) or in_hud(marker):
                    continue
                if marker.size > 10 * mover.size:
                    continue
                d = _gdist(mover.centre, marker.centre)
                if d < 1:
                    continue
                socket = cav_dims(marker) == mover.dims
                prio = 8.0 + (1.5 if socket else 0) \
                    + (1.5 if not socket and len(enc[id(marker)]) >= 4 else 0) \
                    - (1.0 if len(enc[id(mover)]) >= 4 else 0) \
                    + (0.5 if mover.size <= marker.size else 0) - min(d, 60) / 60.0
                add("align", mover, marker.centre, f"same-colour marker {marker!r}", prio, marker.size, marker.cells)
    # (b) SIZE FIT: a frame whose cavity has a mover's dims (or k movers wide)
    frames = [c for c in cl if enc[id(c)] and not in_hud(c)]
    for f in frames:
        cav = enc[id(f)]
        rs = [r for r, _ in cav]; cs_ = [c for _, c in cav]
        ih, iw = max(rs) - min(rs) + 1, max(cs_) - min(cs_) + 1
        ctr = ((min(rs) + max(rs)) / 2.0, (min(cs_) + max(cs_)) / 2.0)
        for m in cl:
            if m.color == f.color or not mover_ok(m) or m.size < 2:
                continue
            if len(enc[id(m)]) > 4:
                continue        # a frame itself is not a mover
            if any(m.cells == enc[id(f2)] for f2 in frames):
                continue        # exactly filling a frame: a fill, not a mover
            mh, mw = m.dims
            if (ih, iw) == (mh, mw) or (ih == mh and iw > mw and iw % mw == 0 and iw // mw <= 4) \
                    or (iw == mw and ih > mh and ih % mh == 0 and ih // mh <= 4):
                d = _gdist(m.centre, ctr)
                prio = 8.5 - min(d, 60) / 60.0 - (0.3 if len(by_col[m.color]) > 1 else 0)
                add("align", m, ctr, f"fits frame {f!r}", prio, f.size, f.cells)
    # (c) SHAPE CONGRUENCE across colours (a piece and a constellation)
    for m in cl:
        if not mover_ok(m) or m.n_parts > 1 or m.size < 3:
            continue
        for t in cl + const:
            if t.color == m.color or t is m or t.n_parts < 3:
                continue
            if t.norm == m.norm:
                add("cover", m, t.cells, f"congruent constellation {t!r}", 7.5, t.size)
    # (e) FIT BY SHAPE: two single things of different colours with the
    #     same shape up to a turn or a flip — one goes onto the other
    #     (which is the mover, the keys will tell; both are read).  A
    #     thing with a twin of its own colour elsewhere is not read this
    #     way (that is a pair, rule a).
    singles = [c for c in cl if c.n_parts == 1 and c.size >= 4 and not in_hud(c) and not border(c)]
    for m in singles:
        if not mover_ok(m):
            continue
        for t in singles:
            if t is m or t.color == m.color or t.size != m.size or t.dims not in (m.dims, m.dims[::-1]):
                continue
            if t.norm in _gdihedral(m.norm) and _gdist(m.centre, t.centre) >= 1:
                turned = t.norm != m.norm
                add("cover", m, t.cells, f"fits the same shape C{t.color}[{t.size}@{t.bbox}]" + (" turned" if turned else ""), 8.3 - (0.2 if turned else 0), t.size)
    # (f) FIT BY MARKS: a pattern of like marks on one body and the same
    #     pattern (up to a turn) on another — the marks go onto the marks
    #     (a plug and its socket, two pieces that dock).  Marks: small
    #     things of one colour touching a larger host of another colour.
    def host_of(mk):
        best = None
        for h in cl:
            if h is mk or h.color == mk.color or h.size <= mk.size:
                continue
            if _gbbox_gap(h.bbox, mk.bbox) <= 1 and any(abs(r - hr) + abs(c - hc) == 1 for r, c in mk.cells for hr, hc in h.cells):
                if best is None or h.size < best.size:
                    best = h
        return best
    marks_by_host = defaultdict(list)
    for mk in cl:
        if mk.size <= 16 and mk.n_parts == 1 and not in_hud(mk):
            h = host_of(mk)
            if h is not None:
                marks_by_host[(id(h), mk.color)].append((h, mk))
    groups = [(h_mks[0][0], [mk for _, mk in h_mks]) for h_mks in marks_by_host.values() if len(h_mks) >= 2]
    for i, (ha, mka) in enumerate(groups):
        for j, (hb, mkb) in enumerate(groups):
            if i == j or ha is hb or len(mka) != len(mkb) or mka[0].color != mkb[0].color:
                continue
            if any(a.norm != mka[0].norm for a in mka) or any(b.norm != mkb[0].norm for b in mkb):
                continue
            ca = set(); cb = set()
            for a in mka:
                ca |= a.cells
            for b in mkb:
                cb |= b.cells
            pa, pb = GoalCluster(mka[0].color, ca, len(mka)), GoalCluster(mkb[0].color, cb, len(mkb))
            if pb.norm in _gdihedral(pa.norm) and mover_ok(ha):
                turned = pb.norm != pa.norm
                add("cover", pa, pb.cells, f"marks onto like marks C{pb.color}[{pb.size}@{pb.bbox}] of C{hb.color}" + (" turned" if turned else ""), 8.6 - (0.2 if turned else 0), pb.size)
    # (d) A TWIN IN THE HUD names the mover; an enclosing marker of a unique
    #     colour is where it goes
    for col, cs in by_col.items():
        if len(cs) != 2 or cs[0].norm != cs[1].norm:
            continue
        a, b = cs
        if in_hud(a) == in_hud(b):
            continue
        mover = b if in_hud(a) else a
        if not mover_ok(mover):
            continue
        for rg in cl:
            if rg.color == col or in_hud(rg) or not enc[id(rg)] or len(by_col[rg.color]) != 1:
                continue
            d = _gdist(mover.centre, rg.centre)
            # a twin in the HUD is the strongest reading a first frame
            # offers: the game itself says what matters
            add("align", mover, rg.centre, f"HUD twin -> ring {rg!r}", 9.3, rg.size, rg.cells)
    hyps.sort(key=lambda h: -h.prio)
    seen, out, per_col = set(), [], Counter()
    for h in hyps:
        key = (h.m_color, h.m_start, h.note)
        if key in seen or per_col[h.m_color] >= 4:
            continue
        seen.add(key)
        per_col[h.m_color] += 1
        out.append(h)
    return out[:12]


class GoalEffectMap:
    """LEVEL-WIDE effect knowledge: for every key executed, what every
    cluster colour did (delta of the nearest same-colour cluster).  One
    exploration pass serves every hypothesis of the level: a candidate
    whose mover no key ever moved is refuted at zero cost.  Records are
    RECENT (the last three), so a change of mode overwrites rather than
    averages."""
    KEEP = 3

    def __init__(self):
        self.effects = defaultdict(lambda: defaultdict(list))   # key -> color -> [delta]
        self.tested = set()
        self.pending = None      # (key, snapshot {color: [centres]}, could_be_mode)
        self.modes = set()       # keys that moved nothing large: mode candidates
        self.mode_presses = 0
        self.retest = []         # after a mode press: keys to re-check, movers first
        self.moved_things = set()   # click keys whose clicked thing moved: not buttons
        self.pending_thing = None
        self.frame = (64, 64)
        self.playfield = None

    def configure(self, frame, playfield):
        self.frame, self.playfield = frame, playfield

    def in_field(self, c):
        """Not a HUD bar (a step counter shrinks at every action) and not
        outside the playfield."""
        fh, fw = self.frame
        if c.dims[1] >= fw - 2 or c.dims[0] >= fh - 2:
            return False
        if self.playfield is not None:
            r0, r1, c0, c1 = self.playfield
            if c.bbox[2] < r0 or c.bbox[0] >= r1 or c.bbox[3] < c0 or c.bbox[1] >= c1:
                return False
        return True

    def snapshot(self, cl):
        snap = defaultdict(list)
        for c in cl:
            if self.in_field(c):
                snap[c.color].append((c.centre, c.size))
        return snap

    def note_action(self, key, cl, could_be_mode=False):
        self.pending = (key, self.snapshot(cl), could_be_mode)
        self.tested.add(key)
        # remember the THING under a click, not its colour: a button that
        # stays put is a button even when its colour moved elsewhere
        # (s5i5 L2: the bracket's colour is the arm's colour)
        if isinstance(key, tuple) and key[0] == "click":
            pt = key[1]
            under = [c for c in cl if c.bbox[0] <= pt[0] <= c.bbox[2] and c.bbox[1] <= pt[1] <= c.bbox[3]]
            self.pending_thing = min(under, key=lambda c: c.size).centre if under else None
        else:
            self.pending_thing = None

    def observe(self, cl):
        if self.pending is None:
            return
        key, before, could_be_mode = self.pending
        self.pending = None
        if getattr(self, "pending_thing", None) is not None:
            # did the clicked thing itself go anywhere?
            pt = self.pending_thing
            still = any(abs(c.centre[0] - pt[0]) + abs(c.centre[1] - pt[1]) < 1 for c in cl)
            if not still:
                self.moved_things.add(key)
            self.pending_thing = None
        after = self.snapshot(cl)
        moved_big = False
        changed = any(sorted(before.get(c, [])) != sorted(after.get(c, []))
                      for c in set(before) | set(after))
        for color, items in before.items():
            # one record per press: the motion of the cluster of this
            # colour that moved most (two dots of one colour are two
            # clusters; the press moved one of them)
            best = (0, 0)
            for (pos, size) in items:
                cands = [(p, s) for p, s in after.get(color, []) if 0.3 * size <= s <= 3.0 * size]
                if not cands:
                    continue
                p2, _ = min(cands, key=lambda ps: _gdist(ps[0], pos))
                dlt = (p2[0] - pos[0], p2[1] - pos[1])
                if abs(dlt[0]) + abs(dlt[1]) > 24:
                    continue    # vanished and re-matched to another: not a motion
                if abs(dlt[0]) + abs(dlt[1]) > abs(best[0]) + abs(best[1]):
                    best = dlt
                if dlt != (0, 0) and size >= 3:
                    moved_big = True
            lst = self.effects[key][color]
            lst.append(best)
            del lst[:-self.KEEP]
        if not moved_big and changed and could_be_mode:
            # something small changed and nothing large moved: a
            # selection mark, a livery — the key may have changed hands
            if key not in self.modes:
                # the press just made IS the mode press: re-check at once
                # what the record says moves things
                self.modes.add(key)
                self.set_retest(key)
        elif moved_big and key in self.modes:
            self.modes.discard(key)

    def mean(self, key, color):
        es = self.effects.get(key, {}).get(color)
        if not es:
            return None
        return (sum(e[0] for e in es) / len(es), sum(e[1] for e in es) / len(es))

    def inert_for(self, key, color):
        es = self.effects.get(key, {}).get(color)
        return bool(es) and len(es) >= 2 and all(e == (0, 0) for e in es[-2:])

    def switch_mode(self):
        """A key that moves nothing large may change WHICH body obeys the
        others (re86's ACTION5, a selection click): press it.  What the
        other keys did is kept as a prior to be re-checked, not erased."""
        if not self.modes:
            return None
        key = sorted(self.modes, key=str)[self.mode_presses % len(self.modes)]
        self.mode_presses += 1
        self.set_retest(key)
        return key

    def set_retest(self, key):
        moved = [k for k, cols in self.effects.items() if k != key
                 and any(e != (0, 0) for es in cols.values() for e in es)]
        self.retest = moved

    def invalidate(self, keep):
        """A key contradicted its record: the mode changed. Everything
        learned before is suspect except the key just re-checked."""
        for k in list(self.effects):
            if k != keep:
                self.effects.pop(k, None)
                self.tested.discard(k)

    def ever_moved(self, color):
        return any(any(e != (0, 0) for e in cols.get(color, []))
                   for cols in self.effects.values())


class GoalActuator:
    """Greedy descent on a hypothesis using the level's GoalEffectMap.  Keys:
    'ACTIONk', ('click', (r, c)) for a point on a thing, ('rclick',
    (dr, dc)) for a point relative to the mover.  The mover is tracked as
    the same-colour cluster nearest its last position with a compatible
    size (and, when known, its shape)."""

    def __init__(self, hyp, actions, emap, budget=40, frame=(64, 64), playfield=None):
        self.h = hyp
        self.actions = [a for a in actions if a not in ("ACTION6", "RESET")]
        self.clickable = "ACTION6" in actions
        self.emap = emap
        self.budget = budget
        self.frame = frame
        self.playfield = playfield
        emap.configure(frame, playfield)
        self.m_pos = None
        self.m_cells = None
        self.m_norm = None
        self.cl = []
        self.explored = 0
        self.mode_tries = 0
        self.own = defaultdict(list)     # key -> [delta of THIS mover]
        self.own_at = {}                 # key -> where the last own record was made
        self.last_key = None
        self.last_pos = None
        self.obj_keys = set()            # click keys that land on a thing
        self.self_clicked = False        # the mover itself is clicked once
        self.recheck = False             # the key in flight re-checks a record
        self.mode_key = None
        self.moved = False               # this mover has moved under this actuator
        self.blocked = False             # a key that moves it did not, here
        self.unblock_tries = 0
        self.unblock_pending = None
        self.best_d = None               # the least distance seen: progress
        self.since_progress = 0
        self.pred_was_gain = False
        self.handoff = []                # a planner's route, when the descent is blocked
        self.pred = None                 # where the key in flight should leave the mover
        self.arrived = False             # the mover reached the target and merged with it

    def _merged_at_pred(self, cl):
        """The mover has touched its marker and the two read as one
        cluster: if some larger cluster of the mover's colour holds the
        cell a known effect predicted, the mover is there."""
        if self.pred is None or self.last_pos is None:
            return None
        pr = (int(round(self.pred[0])), int(round(self.pred[1])))
        for c in cl:
            if c.color == self.h.m_color and c.size > self.h.m_size and pr in c.cells:
                if self.m_cells:
                    dr, dc = self.pred[0] - self.last_pos[0], self.pred[1] - self.last_pos[1]
                    self.m_cells = frozenset((int(round(r + dr)), int(round(c_ + dc)))
                                             for r, c_ in self.m_cells)
                return self.pred
        return None

    def locate(self, cl, objs=None):
        cands = [c for c in cl if c.color == self.h.m_color
                 and 0.7 * self.h.m_size <= c.size <= 1.45 * self.h.m_size]
        if not cands and objs is not None:
            # two things of one colour a pixel apart read as one cluster
            # (ka59: two blocks side by side): fall back to components
            cands = [GoalCluster(o.color, o.cells, 1) for o in objs if o.color == self.h.m_color
                     and 0.7 * self.h.m_size <= o.size <= 1.45 * self.h.m_size]
        if not cands:
            return self._merged_at_pred(cl)
        ref = self.m_pos if self.m_pos is not None else self.h.m_start
        if self.m_norm is not None:
            same = [c for c in cands if c.norm == self.m_norm]
            if same:
                cands = same
        best = min(cands, key=lambda c: _gdist(c.centre, ref))
        if self.m_pos is not None and _gdist(best.centre, ref) > 16:
            # the thing tracked is gone and the nearest of its kind is far:
            # another of its kind, not this mover — unless it is the only
            # thing of its kind on the board (a jump)
            others = [c for c in cl if c.color == self.h.m_color and c is not best
                      and 0.7 * self.h.m_size <= c.size <= 1.45 * self.h.m_size]
            if others or self.h.m_size <= 2:
                return self._merged_at_pred(cl)
        if self.m_norm is None:
            self.m_norm = best.norm
        self.m_cells = best.cells
        return best.centre

    def _axes(self):
        # only THIS mover's motion says which axes are controllable: the
        # colour-pooled map may carry another thing's motion (s5i5: the
        # other tip's slide marked the row axis free and the dot 'arrived')
        seen = [e for es in self.own.values() for e in es if e != (0, 0)]
        if not seen:
            return True, True
        # an axis is written off only once every key has had its turn
        # with this mover (and a few clicks, where clicking is the game)
        if any(a not in self.own for a in self.actions):
            return True, True
        if self.clickable and sum(1 for k in self.own if isinstance(k, tuple)) < 6:
            return True, True
        return any(abs(e[0]) > 0 for e in seen), any(abs(e[1]) > 0 for e in seen)

    def dist(self, pos):
        """ALIGN: distance along the CONTROLLABLE axes — once effects are
        known, an axis no key can change does not count (vc33: 'level
        with the mark' is a one-axis relation).  COVER: how far the
        target cells are from the mover's cells translated to `pos`
        (zero when every marker cell lies under the mover)."""
        if self.h.kind == "cover" and self.m_cells and self._cells_cover():
            dr = pos[0] - self.m_pos[0] if self.m_pos else 0
            dc = pos[1] - self.m_pos[1] if self.m_pos else 0
            cells = [(r + dr, c + dc) for r, c in self.m_cells]
            tot = 0.0
            for t in self.h.target:
                tot += min(abs(t[0] - r) + abs(t[1] - c) for r, c in cells)
            return tot
        t = self.h.target_centre()
        dr, dc = abs(pos[0] - t[0]), abs(pos[1] - t[1])
        ar, ac = self._axes()
        return (dr if ar else 0.0) + (dc if ac else 0.0)

    def _cells_cover(self):
        """Marks that lie WITHIN the mover's extent are covered cell by
        cell (re86: dots on the arms of a cross); marks that frame a
        larger spot (lp85: four corner marks round a tile) are matched by
        centre."""
        rs = [r for r, _ in self.h.target]; cs = [c for _, c in self.h.target]
        th, tw = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        mr = [r for r, _ in self.m_cells]; mc = [c for _, c in self.m_cells]
        mh, mw = max(mr) - min(mr) + 1, max(mc) - min(mc) + 1
        return th <= mh and tw <= mw

    def observe(self, cl, objs=None):
        self.cl = cl
        self.emap.observe(cl)
        if self.emap.retest and self.last_key in self.emap.modes:
            self.mode_key = self.last_key
        pos = self.locate(cl, objs)
        if self.pred is not None and self.last_pos is not None and self.moved \
                and self.dist(self.pred) < 0.5 + _gdist(self.pred, self.last_pos) \
                and (pos is None or _gdist(pos, self.last_pos) > 6):
            # the mover is not where a known effect should have left it —
            # and that place is the marker: it arrived and merged with the
            # marker (s5i5: the dot inside its diamond is one component)
            self.arrived = True
        self.pred = None
        if self.last_key is not None and self.last_pos is not None and pos is not None:
            dlt = (pos[0] - self.last_pos[0], pos[1] - self.last_pos[1])
            if dlt != (0, 0):
                self.moved = True
                self.blocked = False
            elif self.pred_was_gain:
                # the key that should have moved it toward the marker did
                # nothing here: something is in the way — a route, not a
                # straight line, is needed
                self.blocked = True
            prev = self.own.get(self.last_key) or self.emap.effects.get(self.last_key, {}).get(self.h.m_color)
            if prev and self.recheck:
                if (dlt == (0, 0)) != (prev[-1] == (0, 0)):
                    # the record was contradicted after a mode press: the
                    # world changed hands — everything else is re-learned
                    self.emap.invalidate(self.last_key)
                    self.emap.retest = []
                    self.own = defaultdict(list)
                    self.own_at = {}
                elif dlt != (0, 0):
                    # reproduced: the press was nothing
                    self.emap.retest = []
                    if self.mode_key is not None:
                        self.emap.modes.discard(self.mode_key)
            self.recheck = False
            self.own[self.last_key].append(dlt)
            del self.own[self.last_key][:-GoalEffectMap.KEEP]
            self.own_at[self.last_key] = self.last_pos
        self.pred_was_gain = False
        self.last_key = None
        self.m_pos = pos
        if pos is not None:
            d = self.dist(pos)
            if self.best_d is None or d < self.best_d - 1e-6:
                self.best_d = d
                self.since_progress = 0
            else:
                self.since_progress += 1
        return self.m_pos

    def stalled(self):
        """No progress for twenty actions after the keys were tried: the
        reading is not paying; give the rest of the agent its turn."""
        return self.since_progress >= 20 and (self.explored >= 8 or
                                              all(a in self.own for a in self.actions))

    def effect(self, key):
        """This mover's own record first; the level map as a prior."""
        es = self.own.get(key)
        if es:
            return (sum(e[0] for e in es) / len(es), sum(e[1] for e in es) / len(es)), \
                (len(es) >= 2 and all(e == (0, 0) for e in es[-2:]))
        e = self.emap.mean(key, self.h.m_color)
        return e, self.emap.inert_for(key, self.h.m_color)

    def in_field(self, p):
        if self.playfield is None:
            fh, fw = self.frame
            return 1 < p[0] < fh - 2 and 0 <= p[1] < fw
        r0, r1, c0, c1 = self.playfield
        return r0 <= p[0] < r1 and c0 <= p[1] < c1

    def candidates(self):
        """Exploration order is the order a person pokes: the keys; the
        thing itself and points just beyond it toward the marker; the
        marker; the centre of every other small thing (buttons, handles,
        the other piece); then the ends of elongated things."""
        keys = list(self.actions)
        self.obj_keys = set()
        if self.clickable and self.m_pos is not None:
            t = self.h.target_centre()
            m = self.m_pos
            fh, fw = self.frame
            if not self.self_clicked:
                own = ("click", (int(round(m[0])), int(round(m[1]))))
                keys.append(own)
                self.obj_keys.add(own)
            dr, dc = t[0] - m[0], t[1] - m[1]
            n = max(abs(dr), abs(dc), 1e-6)
            for k in (9, 6, 3):
                # the longest reach first: a pull that works at 9 is
                # three times the progress of one at 3
                off = (int(round(k * dr / n)), int(round(k * dc / n)))
                if off != (0, 0) and self.in_field((m[0] + off[0], m[1] + off[1])):
                    keys.append(("rclick", off))
            keys.append(("click", (int(round(t[0])), int(round(t[1])))))
            cnt = Counter(c.color for c in self.cl)
            # buttons do not move: a colour that moved under our own
            # actions (a line to the mover, the mover's centre) is not
            # a thing to press
            scarce = sorted([c for c in self.cl if c.size <= 60 and cnt[c.color] <= 4
                             and c.color != self.h.m_color
                             and ("click", (int(round(c.centre[0])), int(round(c.centre[1])))) not in self.emap.moved_things
                             and not (c.size <= 3 and self.emap.ever_moved(c.color))
                             and not (c.bbox[0] <= 1 or c.bbox[2] >= fh - 2)   # HUD rows
                             and self.in_field(c.centre)
                             and c.dims[1] < fw - 2 and c.dims[0] < fh - 2],
                            key=lambda c: c.size)
            sides = []
            for c in scarce:
                r0, c0, r1, c1 = c.bbox
                cr, cc = c.centre
                k = ("click", (int(round(cr)), int(round(cc))))
                keys.append(k)
                self.obj_keys.add(k)
                if c.dims[0] >= 3:
                    sides += [(r0 + 0.5, cc), (r1 - 0.5, cc)]
                if c.dims[1] >= 3:
                    sides += [(cr, c0 + 0.5), (cr, c1 - 0.5)]
            for p in sides:
                k = ("click", (int(round(p[0])), int(round(p[1]))))
                keys.append(k)
                self.obj_keys.add(k)
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def resolve(self, key):
        """The click point for a chosen key (None for a named action)."""
        if not isinstance(key, tuple):
            return None
        if key[0] == "rclick":
            return (int(round(self.m_pos[0] + key[1][0])), int(round(self.m_pos[1] + key[1][1])))
        return key[1]

    def take(self, key):
        """Bookkeeping for a key chosen by someone else (a planner's
        route): the record still learns from it."""
        self.emap.note_action(key, self.cl, could_be_mode=False)
        self.pred = None
        self.pred_was_gain = False
        self.last_key = key
        self.last_pos = self.m_pos
        self.budget -= 1
        self.h.spent += 1
        return key

    def choose(self):
        if self.m_pos is None or self.budget <= 0:
            return None
        d0 = self.dist(self.m_pos)
        keys = self.candidates()
        best, best_gain = None, 0.0
        for k in keys:
            e, inert = self.effect(k)
            if e is None or inert:
                continue
            gain = d0 - self.dist((self.m_pos[0] + e[0], self.m_pos[1] + e[1]))
            if gain > best_gain + 1e-6:
                best, best_gain = k, gain
        recheck = False
        if best is not None and self.emap.retest:
            recheck = True
        if best is None and self.emap.retest:
            # after a mode press: re-check what the record says — if it
            # still holds, the press was nothing; if not, the world
            # changed hands and the record is rebuilt
            for k in self.emap.retest:
                if k in keys:
                    best = k
                    recheck = True
                    break
            self.emap.retest = [k for k in self.emap.retest if k != best]
        if best is None:
            # explore an untested key — but not forever: twenty per
            # hypothesis, and none if the level's pass already showed no
            # key moves this colour
            if self.explored < 20:
                for k in keys:
                    # a named key is tried once per mover: what did not
                    # move the last mover may move this one
                    untested = (k not in self.own) if not isinstance(k, tuple) \
                        else (k not in self.emap.tested)
                    if untested:
                        best = k
                        self.explored += 1
                        break
        if best is None:
            # a key that did nothing ELSEWHERE may do something here (a
            # block that could not go down where it stood, ka59): one
            # re-probe per key per place
            for k in keys:
                es = self.own.get(k)
                at = self.own_at.get(k)
                if es and all(e == (0, 0) for e in es) and at is not None \
                        and _gdist(at, self.m_pos) >= 3 and not isinstance(k, tuple):
                    best = k
                    break
        if best is None and self.unblock_tries < 4:
            # UNBLOCK: a key that carried the mover before is inert now —
            # a precondition may have run out (vc33: the partner piston
            # has no length left to give). Press a key known to move
            # something ELSE, then try the stalled key again.
            stalled = [k for k in keys if k in self.own
                       and any(e != (0, 0) for e in self.own[k])
                       and all(e == (0, 0) for e in self.own[k][-2:]) and len(self.own[k]) >= 2]
            if stalled:
                if self.unblock_pending is not None and self.unblock_pending in keys:
                    best = self.unblock_pending
                    self.unblock_pending = None
                    del self.own[best][-2:]      # the zeros were the stall, not the key
                else:
                    others = [k for k in keys if k in self.emap.effects
                              and any(any(e != (0, 0) for e in es)
                                      for c, es in self.emap.effects[k].items() if c != self.h.m_color)
                              and not (k in self.own and any(e != (0, 0) for e in self.own[k]))]
                    if others:
                        best = others[self.unblock_tries % len(others)]
                        self.unblock_pending = stalled[0]
                        self.unblock_tries += 1
        if best is None and self.emap.modes and self.mode_tries < 2:
            self.mode_tries += 1
            best = self.emap.switch_mode()
            self.mode_key = best
        if best is None:
            return None
        self.recheck = recheck
        e, _ = self.effect(best)
        self.pred = (self.m_pos[0] + e[0], self.m_pos[1] + e[1]) if e is not None and e != (0, 0) else None
        self.pred_was_gain = best_gain > 0 and self.pred is not None
        if isinstance(best, tuple) and best[0] == "click" and _gdist(best[1], self.m_pos) <= 1.5:
            self.self_clicked = True
        self.emap.note_action(best, self.cl, could_be_mode=(not isinstance(best, tuple)) or best in self.obj_keys)
        self.last_key = best
        self.last_pos = self.m_pos
        self.budget -= 1
        self.h.spent += 1
        return best
