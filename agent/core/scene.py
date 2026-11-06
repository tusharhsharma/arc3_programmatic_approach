"""THE SCENE: persistent objects, and a frame as their rendering.

The component tracker matched pieces of colour between frames and
needed an exception for every way a piece can hide: a tile under a
walker, a tip at the end of a bar that grew over it, a slot behind a
shifted piece, a flash on an edge.  Here the world is a set of OPAQUE
OBJECTS on a background, each with a persistent shape (a colour map)
and an identity, and a frame is what they render to.  Reading a frame
is finding, for every object, the transform that best explains the
pixels (a translation, or none), with the rule that an object may be
HIDDEN by another that lies over it and is then still there; what no
object explains is growth of an object at its edge, a part that
arrived inside an object's box, or a new object; what an object
predicts and the frame does not show, where nothing covers it, is
loss (a bar retracting) or, when nearly everything is gone, the
object vanished.  Occlusion, permanence, a moving hole in a ring and
a flash on an edge are then one rule, not four.

Events are the same closed vocabulary the component tracker emitted:
moved / grew / shrank / recoloured / appeared / vanished / morphed /
covered / uncovered.  Objects expose themselves to the rest of the
core as sprites (cells with colours, body colour, tags) plus their
PARTS (the cells of each non-body colour) as pseudo-sprites with ids
anchored to the object — so a tip that rides its bar keeps its name
for as long as the bar does.
"""
from collections import Counter, defaultdict

import numpy as np

from .entities import Sprite, Event, read_frame, hud_strips, BIG, _rebase
import os
_ENV = dict(os.environ)      # (the switches are read once: a lookup in os.environ is a Mapping call, and one stood in the planner's successor loop — 20 million calls, 60 s of a 300 s level on bp35 L2)
_DEBUG = bool(_ENV.get("SCENE_DEBUG"))


def _paint(cells, shape, bg):
    """A frame of background with these cells painted on it."""
    fr = np.full(shape, bg, dtype=np.int64)
    for (r, c), col in cells.items():
        fr[r, c] = col
    return fr



def _four_pieces(cells):
    """The 4-connected pieces of a set of cells."""
    left = set(cells); out = []
    while left:
        start = left.pop(); comp = {start}; stack = [start]
        while stack:
            r, c = stack.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in left:
                    left.discard(q); comp.add(q); stack.append(q)
        out.append(comp)
    return out


class SceneTracker:
    """Objects that persist; frames that are read against them."""

    def __init__(self):
        self.next_id = 1
        self.objects = {}            # sid -> Sprite (full shape as last fully seen)
        self.hidden = {}             # sid -> Sprite for objects entirely covered now
        self.prev = []               # what the rest of the core sees: sprites + parts
        self.last_events = []
        self.split_keys = set()
        self.history = {}
        self.worn = defaultdict(set)  # sid -> every colour the thing has worn (a docked mark turns green)
        self.ghosts = {}             # sid -> (Sprite as last seen, updates since): a thing gone to nothing keeps its place

    # ------------------------------------------------------------ helpers
    def _new(self, s):
        s.sid = self.next_id; self.next_id += 1
        return s

    @staticmethod
    def _score(cells, frame, v, bg):
        """How many of the object's cells the frame shows at offset v, and
        how many it contradicts where the frame is NOT another colour
        that could be a coverer (i.e. shows background)."""
        H, W = frame.shape
        hit = 0; miss_bg = 0; covered = 0
        for (r, c), col in cells.items():
            rr, cc = r + v[0], c + v[1]
            if rr < 0 or cc < 0 or rr >= H or cc >= W:
                miss_bg += 1; continue
            f = int(frame[rr, cc])
            if f == col:
                hit += 1
            elif f == bg:
                miss_bg += 1
            else:
                covered += 1
        return hit, miss_bg, covered

    @staticmethod
    def _rot_map(cells, k, pivot=None):
        """A colour map turned k quarter-turns clockwise about the corner of
        its box (or of `pivot`'s box), rebased so the box's top-left stays.
        Returns (turned map, the cell map old -> new)."""
        src = pivot if pivot is not None else cells
        r0 = min(r for r, _ in src); c0 = min(c for _, c in src)
        r1 = max(r for r, _ in src); c1 = max(c for _, c in src)
        cur = {p: p for p in cells}
        rr0, cc0, rr1, cc1 = r0, c0, r1, c1
        for _ in range(k % 4):
            h = rr1 - rr0
            f = lambda r, c: (c - cc0 + rr0, h - (r - rr0) + cc0)
            cur = {p: f(*q) for p, q in cur.items()}
            # the pivot box after this turn
            pr = [f(r, c) for r in (rr0, rr1) for c in (cc0, cc1)]
            rr0, cc0 = min(r for r, _ in pr), min(c for _, c in pr)
            rr1, cc1 = max(r for r, _ in pr), max(c for _, c in pr)
        dv = (r0 - rr0, c0 - cc0)
        cur = {p: (q[0] + dv[0], q[1] + dv[1]) for p, q in cur.items()}
        return {cur[p]: col for p, col in cells.items()}, cur

    def _tag_ground(self, frame, bg):
        """A colour that covers a large share of the frame is ground (floor
        or wall): every piece of it is terrain, however small the piece
        (a floor patch between two blocks is floor, not a thing)."""
        area = Counter(); biggest = Counter()
        for o in self.objects.values():
            area[o.body] += o.size
            biggest[o.body] = max(biggest[o.body], o.size)
        # (a floor is a large colour in large pieces; many small blocks of
        # one colour are blocks, however much of the frame they cover)
        self.ground = {c for c, a in area.items() if a >= 300 and biggest[c] >= 100}
        for o in self.objects.values():
            if o.body in self.ground:
                o.tags.add("terrain")

    def _candidates(self, obj, frame, bg, radius=16):
        """Vectors worth trying: those that several of the object's cells
        vote for (a cell of the object's colour lies there), plus zero.
        Terrain does not move."""
        if "terrain" in obj.tags:
            return [(0, 0)]
        votes = Counter()
        cells = list(obj.cells.items())
        step = max(1, len(cells) // 24)
        H, W = frame.shape
        for (r, c), col in cells[::step]:
            r0, r1 = max(0, r - radius), min(H - 1, r + radius)
            c0, c1 = max(0, c - radius), min(W - 1, c + radius)
            sub = frame[r0:r1 + 1, c0:c1 + 1]
            for rr, cc in zip(*np.nonzero(sub == col)):
                votes[(int(rr) + r0 - r, int(cc) + c0 - c)] += 1
        cands = [v for v, _ in votes.most_common(12)]
        # (a shift under which the WHOLE thing is found is a candidate
        # however many like-coloured cells vote for other shifts: among
        # many blocks of one colour a thing that jumped far is still whole
        # somewhere, and the nearest such place is tried first)
        n_s = len(cells[::step])
        whole = sorted((v for v, k in votes.items() if k >= 0.9 * n_s and v not in cands), key=lambda v: abs(v[0]) + abs(v[1]))
        cands += whole[:12]
        if (0, 0) not in cands:
            cands.append((0, 0))
        # (a thing that stepped last frame may step again — by the same
        # step in any of the four directions: those are tried whatever the
        # votes say, a step among many like-coloured things is easily outvoted)
        mv = getattr(obj, "moved_by", None)
        if mv:
            m = max(abs(mv[0]), abs(mv[1]))
            for v in ((m, 0), (-m, 0), (0, m), (0, -m)):
                if v not in cands:
                    cands.append(v)
        hint = getattr(self, "scroll_hint", None)
        if hint is not None and hint not in cands:
            cands.append(hint)              # (the whole screen may have moved: tried by everything)
        return cands

    @staticmethod
    def _global_shift(prev, frame, bg, radius=48, strict=False):
        """The one shift that carries the last frame onto this one (a scene
        that scrolled): the shift under which the overlapping window agrees
        far better than staying put does, and nearly wholly; else None.
        Read first with the background lifted off; when that reads nothing,
        with every cell counted and stricter (a world of two grounds — a
        black maze on blue — has no one background to lift)."""
        H, W = frame.shape
        total = int(((frame != bg) | (prev != bg)).sum())   # (what is drawn in either frame: a room that
        if total == 0:                                      # scrolled off must count against staying put)
            return None
        base = int(((prev == frame) & (frame != bg)).sum()) / float(total)
        if base >= 0.97:
            return None if strict or bg == -1 else SceneTracker._global_shift(prev, frame, -1, radius=radius, strict=True)
        changed = (prev != frame) & (frame != bg)          # what staying put leaves unexplained (of what is drawn now)
        n_changed = int(((prev != frame) & ((frame != bg) | (prev != bg))).sum())
        if n_changed < (0.15 if strict else 0.08) * total or len(set(frame[changed].tolist())) < 2:
            return None if strict or bg == -1 else SceneTracker._global_shift(prev, frame, -1, radius=radius, strict=True)
        best = None
        # (every shift within 24; beyond that only straight ones — a screen
        # that follows a body flying far moves along one axis)
        shifts = [(dr, dc) for dr in range(-24, 25) for dc in range(-24, 25)]
        shifts += [(dr, 0) for dr in range(-radius, radius + 1) if abs(dr) > 24]
        shifts += [(0, dc) for dc in range(-radius, radius + 1) if abs(dc) > 24]
        for dr, dc in shifts:
            if True:
                if dr == 0 and dc == 0:
                    continue
                r0, r1 = max(0, dr), min(H, H + dr)
                c0, c1 = max(0, dc), min(W, W + dc)
                a = prev[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
                b = frame[r0:r1, c0:c1]
                nb = int(((b != bg) | (a != bg)).sum())
                # (a window too small to speak for the screen — unless it
                # speaks almost wholly: a body that flew six cells carries
                # a third of the screen with it, and that third agrees)
                if nb < 0.4 * total and not (nb >= 0.3 * total and int(((a == b) & (b != bg)).sum()) >= 0.9 * nb):
                    continue
                agree = (a == b) & (b != bg)
                frac = int(agree.sum()) / float(nb)
                # ...and of what staying put left unexplained, how much this shift explains
                ch = changed[r0:r1, c0:c1]
                expl = int((agree & ch).sum()) / float(max(1, int(ch.sum())))
                # ...and what stayed put must not contradict it: cells that agree
                # with staying put and not with the shift are things that did
                # not move — many of them, and it was a thing that moved, not
                # the screen (a chain of arms pushed along is no scroll)
                still = (prev[r0:r1, c0:c1] == b) & (b != bg) & ~agree
                static = int(still.sum()) / float(nb)
                if static > 0.15:
                    continue
                if frac < 0.85:
                    continue
                # (among shifts that carry the frame well, the one that explains
                # most of what CHANGED: a periodic texture fits many shifts)
                if best is None or (expl, frac) > best[0]:
                    best = ((expl, frac), (dr, dc))
        need_frac, need_expl = (0.9, 0.9) if strict else (0.85, 0.6)
        if best is not None and best[0][1] >= need_frac and best[0][0] >= need_expl and best[0][1] >= base + 0.1:
            return best[1]
        if not strict and bg != -1:
            return SceneTracker._global_shift(prev, frame, -1, radius=radius, strict=True)
        return None

    # ------------------------------------------------------------ update
    def update(self, frame, bg):
        self.n_updates = getattr(self, "n_updates", 0) + 1
        frame = np.asarray(frame)
        H, W = frame.shape
        events = []
        if not self.objects and not self.hidden:
            for s in read_frame(frame, bg, glue_blocks=False):
                self._new(s)
                self.objects[s.sid] = s
            self._split_units(self.objects, events)
            self._tag_ground(frame, bg)
            self.last_frame = frame.copy()
            self.prev = self._expose()
            self.last_events = events
            return events
        explained = np.zeros(frame.shape, dtype=bool)      # frame cells accounted for
        placed = {}                                        # sid -> (v, cells at v)
        # THE SCREEN MAY HAVE MOVED: one shift that carries the last frame
        # onto this one is offered to every thing as a candidate
        self.scroll_hint = None
        lf = getattr(self, "last_frame", None)
        if lf is not None and lf.shape == frame.shape:
            self.scroll_hint = self._global_shift(lf, frame, bg)
            if _DEBUG and self.scroll_hint:
                print(f"      scroll hint {self.scroll_hint}")
            if self.scroll_hint and _ENV.get("SCROLL_DUMP"):
                k_ = getattr(self, "_dump_n", 0); self._dump_n = k_ + 1
                np.save(f"/tmp/scroll_{k_}.npy", np.stack([lf, frame]))
        self.last_frame = frame.copy()
        # 1. every object proposes its best transform; the most certain
        #    are placed first and take their cells (an object cannot
        #    explain a cell another object already explains)
        proposals = []
        ranked = {}
        all_now = list(self.objects.items()) + list(self.hidden.items())
        turn_maps = {}                                     # sid -> (k, cell map) of its best option when it is a turn
        for sid, obj in all_now:
            n = len(obj.cells)
            opts = []
            for v in self._candidates(obj, frame, bg):
                hit, miss_bg, cov = self._score(obj.cells, frame, v, bg)
                # static bias: staying put wins any tie, and wins
                # outright when the object is mostly still visible
                # (when the whole screen moved, what fits its motion whole moved
                # with it — before staying put, which a periodic texture also fits)
                key = (hit / n + (0.15 if v == (0, 0) else 0.0) + (0.2 if (v == getattr(self, "scroll_hint", None) and hit == n) else 0.0), -abs(v[0]) - abs(v[1]))
                opts.append((key, v, hit, {(r + v[0], c + v[1]): col for (r, c), col in obj.cells.items()}, 0))
            # MOVED AND REDRAWN WITHIN: a thing of several colours whose whole
            # footprint lands, shifted, on cells of its own colours is the
            # thing moved — its colours re-read within (an eye that turned
            # to the other side).  Read before a turn (a twentieth cheaper:
            # what moved is one thing, however it is drawn)
            own_cols = set(obj.cells.values())
            if "terrain" not in obj.tags and "hud" not in obj.tags and len(own_cols) >= 2 and 4 <= n <= 400 and max(o[2] for o in opts) < 0.9 * n:
                for v in self._candidates(obj, frame, bg):
                    if v == (0, 0):
                        continue
                    soft = 0; exact = 0; redrawn = {}
                    for (r, c), col in obj.cells.items():
                        rr, cc = r + v[0], c + v[1]
                        if not (0 <= rr < H and 0 <= cc < W):
                            break
                        f = int(frame[rr, cc])
                        if f not in own_cols:
                            break
                        soft += 1; exact += (f == col); redrawn[(rr, cc)] = f
                    else:
                        if soft == n and exact >= 0.4 * n:
                            opts.append(((1.0 - 0.05, -abs(v[0]) - abs(v[1])), v, n, redrawn, 0))
            # THE SAME SHAPE TURNED IS THE SAME THING: a quarter-turn (or a
            # half) about the corner of its box, wherever it came to rest —
            # read only when it explains the frame better than any shift
            # (a turn costs a tenth: a shift that explains as much wins)
            if "terrain" not in obj.tags and 4 <= n <= 400 and max(o[2] for o in opts) < 0.9 * n:
                for k in (1, 3, 2):
                    turned, cmap = self._rot_map(obj.cells, k)
                    if set(turned) == set(obj.cells) and turned == obj.cells:
                        continue                           # symmetric: a turn is no reading
                    tobj = Sprite(sid=-1, cells=turned, parts=obj.parts, tags=set(obj.tags))
                    for v in self._candidates(tobj, frame, bg):
                        hit, miss_bg, cov = self._score(turned, frame, v, bg)
                        if hit < 0.9 * n:
                            continue
                        key = (hit / n - 0.1, -abs(v[0]) - abs(v[1]))
                        opts.append((key, v, hit, {(r + v[0], c + v[1]): col for (r, c), col in turned.items()}, k))
            # A FAR JUMP: a thing gone from where it was (nothing near
            # explains half of it) and standing whole somewhere else — a
            # teleport, a put-back — is read there, before anything nearer
            # takes those cells for a growth of its own
            if "terrain" not in obj.tags and "hud" not in obj.tags and 4 <= n <= 400 and max(o[2] for o in opts) < 0.5 * n:
                for v in self._candidates(obj, frame, bg, radius=64):
                    hit, miss_bg, cov = self._score(obj.cells, frame, v, bg)
                    if hit == n:
                        key = (1.0 + 0.05, -abs(v[0]) - abs(v[1]))      # certain: whole, wherever it is
                        opts.append((key, v, hit, {(r + v[0], c + v[1]): col for (r, c), col in obj.cells.items()}, 0))
            # RECOLOURED WHERE IT STANDS: a thing whose whole footprint shows
            # one other colour (a piece drawn in the selected colour, a tile
            # that lit) did not go anywhere — it is the same thing in a new
            # colour; read before any shift that explains less than all of it
            # (a piece must not jump to the like-coloured piece across the
            # board when the selection colour passes from one to the other)
            if "terrain" not in obj.tags and "hud" not in obj.tags and 4 <= n <= 400 and max(o[2] for o in opts) < n:
                cols_here = set()
                for (r, c) in obj.cells:
                    if not (0 <= r < H and 0 <= c < W):
                        cols_here = None; break
                    cols_here.add(int(frame[r, c]))
                if cols_here and len(cols_here) == 1 and next(iter(cols_here)) != bg:
                    nc_ = next(iter(cols_here))
                    opts.append(((1.0 - 0.02, 0), (0, 0), n, {p: nc_ for p in obj.cells}, 0))
                elif cols_here and bg not in cols_here and len(cols_here) >= 2:
                    # (partly under another thing: the part that shows is one
                    # other colour on most of it — recoloured, and still covered)
                    cnt_ = Counter(int(frame[r, c]) for (r, c) in obj.cells)
                    nc_, k_ = cnt_.most_common(1)[0]
                    if k_ >= 0.5 * n and nc_ not in own_cols:
                        opts.append(((k_ / n - 0.1, 0), (0, 0), k_, {p: nc_ for p in obj.cells if int(frame[p[0], p[1]]) == nc_}, 0))
            opts.sort(key=lambda o: o[0], reverse=True)
            ranked[sid] = opts
        # TWO THINGS THAT EXCHANGED PLACES EXCHANGED COLOURS: when A, whole,
        # could be read as moved onto B's exact cells and B, whole, as moved
        # onto A's — the same shape, each now wearing the other's colour —
        # nothing passed through anything: both recoloured where they
        # stand (sp80 L2: a click makes the clicked bar wear the chosen
        # colour and the chosen bar the plain one; read as the chosen bar
        # jumping, every law after it was of a bar that never moved).  A
        # walker that left a tile is no exchange: its old cells wear the
        # ground, not the tile's colour.
        swapped_ = set()
        if not _ENV.get("NO_SWAP_RULE"):
            objs_ = [(sid_, obj_) for sid_, obj_ in all_now
                     if 4 <= len(obj_.cells) <= 400 and "terrain" not in obj_.tags and "hud" not in obj_.tags]
            by_shape_ = {}
            for sid_, obj_ in objs_:
                by_shape_.setdefault((len(obj_.cells), obj_.shape), []).append((sid_, obj_))
            # (a thing whose cells now show the colour of a thing known to lie
            # UNDER it has uncovered that thing by leaving, not exchanged
            # colours with anyone — tu93's walker leaving a node)
            def uncovers_(obj_, worn_cols_):
                cs_ = set(obj_.cells.keys())
                for h_ in self.hidden.values():
                    if set(h_.cells.keys()) & cs_ and set(h_.cells.values()) & worn_cols_:
                        return True
                return False
            # (ONLY UNDER A CLICK: a colour that passes to the thing pointed at
            # is a choice; a colour that passes to the next thing on a key is
            # the thing's own step — tu93's walker hops from node to node and
            # its old node shows through: the frames alone cannot tell the
            # two apart, the pointer can)
            click_ = getattr(self, "last_click", None)
            by_shape_ = by_shape_ if click_ is not None else {}
            for group_ in by_shape_.values():
                if len(group_) < 2:
                    continue
                if not any(obj_.bbox[0] <= click_[0] <= obj_.bbox[2] and obj_.bbox[1] <= click_[1] <= obj_.bbox[3] for _, obj_ in group_):
                    continue
                worn_ = {}
                for sid_, obj_ in group_:
                    try:
                        worn_[sid_] = {int(frame[r, c]) for (r, c) in obj_.cells}
                    except IndexError:
                        worn_[sid_] = None
                for ia_, (sid_a, obj_a) in enumerate(group_):
                    if sid_a in swapped_ or worn_[sid_a] is None:
                        continue
                    own_a = set(obj_a.cells.values())
                    if worn_[sid_a] == own_a:
                        continue                        # still in its own colours: no exchange
                    if uncovers_(obj_a, worn_[sid_a]):
                        continue
                    for sid_b, obj_b in group_[ia_ + 1:]:
                        if sid_b in swapped_ or worn_[sid_b] is None:
                            continue
                        own_b = set(obj_b.cells.values())
                        if worn_[sid_b] == own_b or own_a == own_b:
                            continue
                        if worn_[sid_a] == own_b and worn_[sid_b] == own_a and not uncovers_(obj_b, worn_[sid_b]):
                            for sid_x, obj_x in ((sid_a, obj_a), (sid_b, obj_b)):
                                new_map_ = {p_: int(frame[p_[0], p_[1]]) for p_ in obj_x.cells}
                                ranked[sid_x].insert(0, ((1.0 + 0.06, 0), (0, 0), len(obj_x.cells), new_map_, 0))
                                swapped_.add(sid_x)
                            if _DEBUG:
                                print(f"      swap: S{sid_a} and S{sid_b} exchanged colours in place")
                            break
        for sid, obj in all_now:
            opts = ranked[sid]
            n = len(obj.cells)
            own_cols = set(obj.cells.values())
            # (among whole readings the larger thing places first: its cells
            # are not a smaller thing's growth)
            # (WHAT IS WHOLE WHERE IT WAS, STAYS: a thing found entire at its
            # own place is placed before any thing that would move onto it —
            # a box among identical boxes does not shuffle along when a
            # neighbour vanishes)
            # (A THING IN ITS OWN COLOURS BEFORE A THING RECOLOURED: a reading
            # that keeps every colour outranks one that must suppose a
            # recolouring — the pad under the arriving walker is covered, not
            # recoloured to the walker's colour; twist suite tw02, where the
            # walker's identity slid onto a far look-alike.  0.02: a tie-break
            # only, below every real cell of difference)
            # (2026-10-16: this tie-break and the "recoloured where it stands"
            # option above were turned every way for cn04 L2 — the piece drawn in
            # the selection colour read as a look-alike across the board.  Every
            # variant that won cn04 L2 lost tu93 L2 or s5i5 L2 or bp35 L1; this
            # one keeps the most.  The statement that tells the recoloured piece
            # from the walker that left a tile is still to find.)
            own_read = set(opts[0][3].values()) <= own_cols
            is_swap_ = sid in swapped_ if not _ENV.get("NO_SWAP_RULE") else False
            stays = 1 if ((opts[0][1] == (0, 0) and opts[0][4] == 0 and opts[0][2] == n and own_read) or is_swap_) else 0
            proposals.append((opts[0][2] / n - (0.0 if (own_read or is_swap_) else 0.02) + (0.06 if is_swap_ else 0.0), stays, opts[0][2] if opts[0][2] == n else 0, opts[0][0][1], sid))
            if opts[0][4]:
                turn_maps[sid] = (opts[0][4], opts[0][1])
        # WHAT CHANGES TOGETHER IS ONE THING: touching small things whose
        # own readings are poor (or a turn) are tried as ONE footprint
        # shifted; when the whole footprint lands on the group's own
        # colours, the group moved by that shift and each member is re-read
        # as the cells of its colour there (a body and the eye beside it
        # that swapped sides).  (Tushar: "what changes in your movement is
        # a full object, even if the same thing changes colour")
        small_ids = [sid for sid, obj in all_now if "terrain" not in obj.tags and "hud" not in obj.tags and 1 <= len(obj.cells) <= 60]
        objs_by = {sid: obj for sid, obj in all_now}
        poor = {sid for sid in small_ids if ranked[sid][0][4] or ranked[sid][0][2] < 0.9 * len(objs_by[sid].cells)}
        if poor:
            # groups of touching small things (8-neighbours), each with a poor member
            parent = {sid: sid for sid in small_ids}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            cell_owner = {}
            for sid in small_ids:
                for p in objs_by[sid].cells:
                    cell_owner[p] = sid
            for sid in small_ids:
                for (r, c) in objs_by[sid].cells:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            o = cell_owner.get((r + dr, c + dc))
                            if o is not None and o != sid:
                                parent[find(sid)] = find(o)
            groups = defaultdict(list)
            for sid in small_ids:
                groups[find(sid)].append(sid)
            for members in groups.values():
                # (what is whole where it was, stays: a thing found entire at
                # its own place is not dragged along by what moved off it —
                # the pen a walker stepped off does not shift with the walker)
                members = [m for m in members if not (ranked[m][0][1] == (0, 0) and ranked[m][0][4] == 0 and ranked[m][0][2] == len(objs_by[m].cells))]
                if len(members) < 2 or not any(m in poor for m in members):
                    continue
                U = {}
                for m in members:
                    U.update(objs_by[m].cells)
                cols = set(U.values())
                if len(cols) < 2 or len(U) > 120:
                    continue
                if sum(len(set(objs_by[m].cells.values())) for m in members) != len(cols):
                    continue                # (two members of one colour: which cells are whose is not read here)
                uobj = Sprite(sid=-1, cells=U, parts=1, tags=set())
                best = None
                for v in self._candidates(uobj, frame, bg):
                    if v == (0, 0):
                        continue
                    soft = exact = 0; ok = True
                    for (r, c), col in U.items():
                        rr, cc = r + v[0], c + v[1]
                        if not (0 <= rr < H and 0 <= cc < W) or int(frame[rr, cc]) not in cols:
                            ok = False; break
                        exact += int(frame[rr, cc]) == col
                    if ok and exact >= 0.4 * len(U):
                        key = (exact, -abs(v[0]) - abs(v[1]))
                        if best is None or key > best[0]:
                            best = (key, v)
                if best is None:
                    continue
                v = best[1]
                for m in members:
                    mcols = set(objs_by[m].cells.values())
                    new = {(r + v[0], c + v[1]): int(frame[r + v[0], c + v[1]]) for (r, c) in U
                           if int(frame[r + v[0], c + v[1]]) in mcols}
                    # (a colour two members share is split by nearness to each member's old cells)
                    if not new:
                        continue
                    ranked[m].insert(0, ((1.0 - 0.05, -abs(v[0]) - abs(v[1])), v, len(new), new, 0))
                    turn_maps.pop(m, None)
                if _DEBUG:
                    print(f"      group {members} moved as one by {v}")
        # WHAT TOUCHES A THING GOES WITH IT: a small thing touching a larger
        # one whose best reading is a shift or a turn is first tried where
        # that shift or turn sends it, when the frame shows it there (two
        # identical stubs on a piece would otherwise swap places)
        rider_sids = set()
        turned_riders = set()
        ride_taken = []             # destinations already ridden to (a hidden mark under a rider does not ride too)
        for sid, obj in all_now:
            if sid in turn_maps or "terrain" in obj.tags or len(obj.cells) > 60:
                continue
            if sid in self.hidden and any(len(set(obj.cells) & src_) >= 0.5 * len(obj.cells) for src_, _ in ride_taken):
                continue            # (what lies hidden under a rider stays where it is: the rider rode, not it)
            for tsid, tobj in all_now:
                if tsid == sid or "terrain" in tobj.tags or len(tobj.cells) <= len(obj.cells):
                    continue
                top = ranked[tsid][0]
                if top[4] == 0 and top[1] == (0, 0):
                    continue
                # (touching, or a cell's gap away: a stub on a notched outline)
                if not any((r + dr, c + dc) in tobj.cells for (r, c) in obj.cells for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2)):
                    continue
                # (a thing still whole where it was did not ride: a wall's
                # dot two cells from a walker stays a wall's dot)
                # (...unless the thing it touches TURNED: a stub whole where
                # it was beside a turned piece is another stub in its place —
                # its own went round with the piece; the ride is tried first)
                if not top[4] and self._score(obj.cells, frame, (0, 0), bg)[0] == len(obj.cells):
                    continue
                v = top[1]
                n = len(obj.cells)
                if top[4]:
                    # a turn about the thing's own box — or about the box of
                    # the WHOLE, thing and stub together (a piece turns with
                    # its marks as one sprite); whichever the frame shows
                    whole = dict(tobj.cells); whole.update(obj.cells)
                    _, cmap = self._rot_map(whole, top[4], pivot=tobj.cells)
                    tries = [{(cmap[p][0] + v[0], cmap[p][1] + v[1]): col for p, col in obj.cells.items()}]
                    _, cmap2 = self._rot_map(whole, top[4], pivot=whole)
                    hn = [cmap2[p] for p in tobj.cells]
                    tb = (min(r for r, _ in top[3]), min(c for _, c in top[3]))
                    v2 = (tb[0] - min(r for r, _ in hn), tb[1] - min(c for _, c in hn))
                    tries.append({(cmap2[p][0] + v2[0], cmap2[p][1] + v2[1]): col for p, col in obj.cells.items()})
                    moved = max(tries, key=lambda mv: sum(1 for (r, c), col in mv.items() if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col))
                else:
                    moved = {(r + v[0], c + v[1]): col for (r, c), col in obj.cells.items()}
                hit = sum(1 for (r, c), col in moved.items() if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col)
                # A RIDER THAT CHANGED COLOUR WHERE IT LANDED: the ride shows
                # one other colour over the whole footprint (a mark that
                # docked and turned green; a mark wearing a colour it wore
                # before) — it rode, and is re-read in that colour
                if hit < 0.9 * n and len(moved) == n:
                    cols_ = {int(frame[r, c]) if 0 <= r < H and 0 <= c < W else -1 for (r, c) in moved}
                    if len(cols_) == 1:
                        nc_ = next(iter(cols_))
                        if nc_ != bg and nc_ >= 0 and nc_ not in set(tobj.cells.values()):
                            moved = {p: nc_ for p in moved}
                            hit = n
                if _DEBUG:
                    print(f"      rider S{sid} with S{tsid} top={top[1]} turn={top[4]}: hit {hit}/{n} at {min(moved) if moved else None}")
                if hit >= 0.9 * n:
                    b0 = obj.bbox; b1 = (min(r for r, _ in moved), min(c for _, c in moved))
                    vv = (b1[0] - b0[0], b1[1] - b0[1])
                    ranked[sid].insert(0, ((hit / n + 0.2, 0), vv, hit, moved, 0))
                    rider_sids.add(sid)
                    ride_taken.append((set(obj.cells), set(moved)))
                    if top[4] and vv != (0, 0):
                        turned_riders.add(sid)
                break
        # (a stub that rode a turn is not a stayer, whole where it was or not;
        # a rider's ride is placed before any other thing's shift onto those
        # cells — a hidden mark does not jump into a stub's place)
        # LIKE THINGS KEEP THEIR NAMES BY THE LEAST MOTION, TOGETHER: when two
        # or more things of one colour each read whole in several places and
        # their first readings collide (one's destination is the other's),
        # the names are given jointly — the assignment of whole readings,
        # one per thing, on cells no two share, that moves them least in all.
        # (tw01: two blocks pushed right by one step each; the lower one,
        # proposed first by the order of its id, took the upper one's new
        # place — four cells away — and the upper one was then found whole
        # eight cells away.  Least motion in all: 4+4 against 12+4.)
        if not _ENV.get("NO_JOINT_NAMES"):
            try:
                import itertools as _it
                by_col_ = defaultdict(list)
                for sid_, obj_ in all_now:
                    if "terrain" in obj_.tags or "hud" in obj_.tags or sid_ in rider_sids or sid_ in turn_maps:
                        continue
                    cols_ = set(obj_.cells.values())
                    if len(cols_) != 1 or not (4 <= len(obj_.cells) <= 400):
                        continue
                    by_col_[next(iter(cols_))].append(sid_)
                for col_, sids_ in by_col_.items():
                    if len(sids_) < 2 or len(sids_) > 6:
                        continue
                    whole_ = {}
                    for sid_ in sids_:
                        n_ = len(objs_by[sid_].cells) if sid_ in objs_by else len((self.objects.get(sid_) or self.hidden.get(sid_)).cells)
                        ws_ = [(o_[1], frozenset(o_[3])) for o_ in ranked[sid_] if o_[2] == n_ and o_[4] == 0 and set(o_[3].values()) == {col_}]
                        if ws_:
                            whole_[sid_] = ws_[:4]
                    if len(whole_) < 2:
                        continue
                    tops_ = {sid_: whole_[sid_][0][1] for sid_ in whole_}
                    clash_ = any(tops_[a_] & tops_[b_] for a_, b_ in _it.combinations(whole_, 2))
                    if not clash_:
                        continue
                    best_ = None
                    order_ = list(whole_)
                    for combo_ in _it.product(*[whole_[sid_] for sid_ in order_]):
                        cells_ = [c_ for _, c_ in combo_]
                        if any(cells_[i_] & cells_[j_] for i_ in range(len(cells_)) for j_ in range(i_ + 1, len(cells_))):
                            continue
                        cost_ = sum(abs(v_[0]) + abs(v_[1]) for v_, _ in combo_)
                        movers_ = sum(1 for v_, _ in combo_ if v_ != (0, 0))
                        if best_ is None or (cost_, movers_) < best_[0]:
                            best_ = ((cost_, movers_), combo_)
                    if best_ is None:
                        continue
                    for sid_, (v_, _) in zip(order_, best_[1]):
                        i_ = next(i2_ for i2_, o_ in enumerate(ranked[sid_]) if o_[1] == v_ and o_[4] == 0 and frozenset(o_[3]) == whole_[sid_][[w_[0] for w_ in whole_[sid_]].index(v_)][1])
                        if i_ != 0:
                            o_ = ranked[sid_].pop(i_)
                            ranked[sid_].insert(0, o_)
                            if _DEBUG:
                                print(f"      S{sid_} named jointly with its like things: {v_} (least motion in all)")
            except Exception as ex_:
                if _DEBUG:
                    print(f"      (joint naming failed: {ex_!r})")
        proposals = [((ranked[sid_][0][0][0] if sid_ in rider_sids else a_), 0 if sid_ in turned_riders else b_, c_, d_, sid_)
                     for (a_, b_, c_, d_, sid_) in proposals]
        proposals.sort(reverse=True)
        for frac, _, _, _, sid in proposals:
            obj = self.objects.get(sid) or self.hidden.get(sid)
            n = len(obj.cells)
            placed[sid] = None
            # the best transform whose cells nobody has taken (a twin that
            # stands where this thing would land was placed first)
            for key, v, hit, shifted, turn in ranked[sid]:
                free_hits = sum(1 for (r, c), col in shifted.items()
                                if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col and not explained[r, c])
                if _DEBUG:
                    print(f"      place S{sid} n={n} try v={v} turn={turn} hit={hit} free={free_hits}")
                # (a thing found whole where it was stays there even when
                # another reading — a part of the same piece — already
                # explains those cells: what did not move does not jump)
                # (...whole IN ITS OWN COLOURS: a reading that supposes a
                # recolouring on cells another thing already explains is that
                # other thing standing there, not this one recoloured)
                if free_hits >= max(1, 0.5 * n) or (v == (0, 0) and free_hits >= 0.3 * n and hit == free_hits) \
                        or (v == (0, 0) and turn == 0 and hit == n and set(shifted.values()) <= set(obj.cells.values())):
                    placed[sid] = (v, shifted, turn)
                    for (r, c), col in shifted.items():
                        if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col:
                            explained[r, c] = True
                    break
            # CUT CLEANLY FROM ONE SIDE, IT IS STILL THE THING: what remains of a
            # thing at its own place, in its own colour, as the thing with one
            # side cut away (a meter spent to its last cell, a bar emptied in
            # one go) is that thing retracted, however much was cut — not a
            # new thing born where the old one vanished (twist suite tw02: the
            # budget bar drained by a pad was reborn under a new number every
            # time, so it was never seen to change under every key)
            if placed[sid] is None and n >= 2:
                left = {(r, c): col for (r, c), col in obj.cells.items()
                        if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col and not explained[r, c]}
                if 1 <= len(left) < n:
                    from .planner import effect_of
                    e_ = effect_of(Sprite(sid=-1, cells=dict(obj.cells), parts=1), Sprite(sid=-1, cells=left, parts=1))
                    if e_[0] == "retract":
                        if _DEBUG:
                            print(f"      place S{sid} n={n} cut from one side to {len(left)} cells ({e_[1]})")
                        placed[sid] = ((0, 0), left, 0)
                        for (r, c) in left:
                            explained[r, c] = True
            # EVERY NEAR PLACE TAKEN: a thing whose near readings all fall on
            # cells another thing already explains (a block that landed
            # beside a like block, whose cells read as the block's own) is
            # looked for WHOLE anywhere, on cells nobody has taken
            if placed[sid] is None and "terrain" not in obj.tags and "hud" not in obj.tags and 4 <= n <= 400:
                for v in self._candidates(obj, frame, bg, radius=64):
                    if v == (0, 0):
                        continue
                    shifted = {(r + v[0], c + v[1]): col for (r, c), col in obj.cells.items()}
                    free_hits = sum(1 for (r, c), col in shifted.items()
                                    if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col and not explained[r, c])
                    if free_hits == n:
                        if _DEBUG:
                            print(f"      place S{sid} n={n} whole and free at far v={v}")
                        placed[sid] = (v, shifted, 0)
                        for (r, c) in shifted:
                            explained[r, c] = True
                        break
        # 1a. RECOLOURED IN PLACE: an object no shift explains whose whole
        #    footprint shows one other colour nobody else explains did not
        #    go anywhere — it changed colour where it stands (a stub that
        #    turned green on docking, a tile that lit)
        for sid, obj in all_now:
            if placed.get(sid) is not None or "terrain" in obj.tags or len(obj.cells) > 400:
                continue
            cols = set()
            ok = True
            for (r, c), col in obj.cells.items():
                if not (0 <= r < H and 0 <= c < W) or explained[r, c] or int(frame[r, c]) == bg:
                    ok = False; break
                cols.add(int(frame[r, c]))
            if ok and len(cols) == 1:
                placed[sid] = ((0, 0), dict(obj.cells), 0)     # (the re-read takes its new colour)
        # 1b. OPAQUE OCCLUSION IS A COMPLETE EXPLANATION: an object placed
        #    as moved, whose old cells all lie under other placed objects
        #    that are not terrain, did not move — it is hidden there (a
        #    tile under a walker), and the cells it claimed elsewhere are
        #    someone else's (a tile the walker left)
        def cover_map_now():
            cm = np.zeros(frame.shape, dtype=bool)
            for sid_, pl_ in placed.items():
                # only a thing that MOVED can have newly covered another:
                # what stood still and shows was under, not over
                if pl_ is None or pl_[0] == (0, 0):
                    continue
                o_ = self.objects.get(sid_) or self.hidden.get(sid_)
                if "terrain" in o_.tags:
                    continue
                for (r, c), col in pl_[1].items():
                    if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col:
                        cm[r, c] = True
            return cm
        # the thing with twins yields first (a tile among tiles is the one
        # that stayed; the one walker is the one that moved); after each
        # reversion the cover is recomputed, so a mover whose old cells
        # were claimed only by the reverted twin keeps its move
        all_objs = {**self.objects, **self.hidden}
        key_n = Counter(o.key for o in all_objs.values())
        order = sorted([sid for sid, pl in placed.items() if pl is not None and pl[0] != (0, 0)],
                       key=lambda sid: (-key_n[all_objs[sid].key], all_objs[sid].size))
        vec_n = Counter(pl[0] for pl in placed.values() if pl is not None and pl[0] != (0, 0))
        for sid in order:
            pl = placed[sid]
            obj = all_objs[sid]
            if vec_n[pl[0]] >= 3 or (sid in rider_sids and pl[1] is ranked[sid][0][3]):
                continue        # a shift that others rode too, or a rider on its thing: it moved with it
            if _DEBUG:
                print(f"      occlusion check S{sid} placed v={pl[0]} rider {sid in rider_sids} same {pl[1] is ranked[sid][0][3]}")
            cm = cover_map_now()
            if all(0 <= r < H and 0 <= c < W and cm[r, c] and int(frame[r, c]) != col for (r, c), col in obj.cells.items()):
                # (PUSHED, NOT HIDDEN: a thing whose moved reading lands on
                # cells nothing else explains did move — a block the walker
                # pushed ahead; read as hidden under the walker, a new block
                # would be born where it went, every press.  Hidden is a
                # reading only when where it would have gone is someone
                # else's — twist suite tw04)
                # (...and nothing HIDDEN would reappear there: the tile a walker
                # steps off is the hidden tile uncovered, not the tile ahead
                # moved backwards — tu93, a board of like tiles)
                dest_ = [(r, c) for (r, c), col in pl[1].items() if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col]
                vacated_ = set()
                for sid_, pl_ in placed.items():
                    if sid_ != sid and pl_ is not None and pl_[0] != (0, 0):
                        vacated_ |= set(all_objs[sid_].cells)
                uncovered_ = any(sid_ != sid and all((r, c) in h_.cells and h_.cells[(r, c)] == int(frame[r, c]) for (r, c) in dest_)
                                 for sid_, h_ in self.hidden.items()) \
                    or all((r, c) in vacated_ for (r, c) in dest_)      # (it "went" where the walker just stood: uncovered, not moved)
                # (...and the thing is seen WHOLE where it went: a pushed block
                # shows every cell at its new place; a reading that matches part
                # of the thing somewhere else is no push — ls20 L2, a glyph
                # "moved" nine rows and five columns on five of its eight cells)
                whole_ = len(dest_) == len(obj.cells)
                if dest_ and whole_ and not uncovered_ and not any(any((r, c) in pl_[1] and pl_[1][(r, c)] == int(frame[r, c]) for (r, c) in dest_)
                                                                   for sid_, pl_ in placed.items() if sid_ != sid and pl_ is not None):
                    if _DEBUG:
                        print(f"      S{sid} moved by {pl[0]}: where it went is nobody else's")
                    continue
                placed[sid] = None
                for (r, c), col in pl[1].items():
                    if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col:
                        explained[r, c] = False
        # 2. objects not placed: hidden under what was placed (all their
        #    cells lie under explained cells of other colours) or gone
        new_objects = {}
        new_hidden = {}
        # Z-ORDER for the re-read: the small and the movable lie on top;
        # terrain is under everything.  What lies on top claims a changed
        # cell as its own recolour; what lies under keeps its colour there
        # (hidden)
        all_objs = {**self.objects, **self.hidden}
        over_map = np.zeros(frame.shape, dtype=bool)
        for sid_, pl_ in placed.items():
            if pl_ is None or "terrain" in all_objs[sid_].tags:
                continue
            for (r, c), col in pl_[1].items():
                if 0 <= r < H and 0 <= c < W and int(frame[r, c]) == col:
                    over_map[r, c] = True
        z_order = sorted(all_objs, key=lambda sid: (1 if "terrain" in all_objs[sid].tags else 0, all_objs[sid].size))
        unplaced = []
        for sid in z_order:
            obj = all_objs[sid]
            was_hidden = sid in self.hidden
            pl = placed.get(sid)
            if pl is None:
                unplaced.append(sid)
                continue
            v, shifted, turn = pl
            if was_hidden:
                events.append(Event("uncovered", sid, {}))
            if turn:
                events.append(Event("turned", sid, {"k": turn, "vec": (float(v[0]), float(v[1]))}))
            # 3. re-read the object where it stands: keep cells the frame
            #    confirms, drop cells the frame shows as background (loss)
            #    keep cells another object covers (still there, hidden)
            kept = {}
            lost = 0
            for (r, c), col in shifted.items():
                if not (0 <= r < H and 0 <= c < W):
                    lost += 1; continue
                f = int(frame[r, c])
                if f == col:
                    kept[(r, c)] = col
                elif f == bg:
                    lost += 1
                else:
                    # another colour: a coverer, or the object recoloured
                    # here (a flash, a mark that changed) — recoloured
                    # when the cell is not explained by anyone else
                    if explained[r, c] and ("terrain" in obj.tags or "texture" in obj.tags):
                        kept[(r, c)] = col          # ground — or a material — under a thing: still there
                    elif explained[r, c]:
                        # a THING partly under another: read as loss (a bar
                        # whose end its neighbour moved onto had retracted);
                        # a thing wholly under another is hidden (below)
                        lost += 1
                    elif "terrain" in obj.tags:
                        pass                        # ground never recolours: what
                                                    # shows on it is a thing (residual)
                    else:
                        kept[(r, c)] = f            # its own new colour
                        explained[r, c] = True
            if not kept:
                events.append(Event("vanished", sid, {"size": obj.size, "body": obj.body}))
                continue
            # a thing whose every cell now wears a GROUND colour is gone (a
            # ring used up shows floor where it was), not recoloured
            ground_ = getattr(self, "ground", set())
            if "terrain" not in obj.tags and obj.body not in ground_ and ground_ and all(col in ground_ for col in kept.values()):
                events.append(Event("vanished", sid, {"size": obj.size, "body": obj.body}))
                continue
            ns = Sprite(sid=sid, cells=kept, parts=obj.parts, tags=set(obj.tags))
            # (the whole thing's motion this frame, for the laws: a thing
            # moved and redrawn within is the thing translated)
            ns.moved_by = tuple(v) if (v != (0, 0) and not turn) else None
            new_objects[sid] = ns
            self.worn[sid] |= set(kept.values())
            if v != (0, 0):
                events.append(Event("moved", sid, {"vec": (float(v[0]), float(v[1]))}))
            recol = sum(1 for p, col in kept.items() if (obj.cells.get(p) if (v == (0, 0) and not turn) else shifted.get(p)) != col)
            if recol:
                events.append(Event("recoloured", sid, {"n": recol, "livery": sorted(set(kept.values()))}))
            if lost:
                events.append(Event("shrank", sid, {"delta": -lost, "vec": (float(v[0]), float(v[1]))}))
            if "terrain" not in obj.tags:
                for (r, c), col in kept.items():
                    if int(frame[r, c]) == col:
                        over_map[r, c] = True       # its full footprint, recolours included
        # the unplaced: hidden under a placed thing that is not terrain
        # (ground showing where a thing stood means it left), else gone
        for sid in unplaced:
            obj = all_objs[sid]
            was_hidden = sid in self.hidden
            cells = obj.cells
            under = all(0 <= r < H and 0 <= c < W and over_map[r, c] and int(frame[r, c]) != col
                        for (r, c), col in cells.items())
            if under and len(cells) <= 400:
                new_hidden[sid] = obj
                if not was_hidden:
                    events.append(Event("covered", sid, {"size": obj.size, "body": obj.body}))
            else:
                events.append(Event("vanished", sid, {"size": obj.size, "body": obj.body}))
        # 4. the residual: frame cells no object explains
        vacated = set()
        for sid_, pl_ in placed.items():
            if pl_ is not None and pl_[0] != (0, 0):
                o_ = all_objs[sid_]
                vacated |= set(o_.cells) - set(pl_[1])
        residual = np.array(frame)
        residual[explained] = bg
        res_sprites = read_frame(residual, bg, glue_blocks=False)
        if _DEBUG:
            print(f"      residual: {[(rs.body, rs.size, rs.bbox, dict(rs.colors)) for rs in res_sprites][:12]}")
        # growth first: a residual piece of an object's own body colour
        # touching that object is the object growing (a bar extending);
        # a tiny residual piece inside an object's box touching it is a
        # part that arrived (a tip at a bar's new end)
        for mode in ("body", "part", "body", "part"):
            for rs in list(res_sprites):
                host = None
                for sid, ns in new_objects.items():
                    if "hud" in ns.tags or (mode == "part" and "terrain" in ns.tags):
                        continue
                    touch = any((r + dr, c + dc) in ns.cells for (r, c) in rs.cells for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                    if not touch:
                        continue
                    if mode == "body":
                        # a wall that grows is a piston: terrain grows too
                        ok = rs.body == ns.body and rs.size <= max(12, ns.size)
                    else:
                        hb = ns.bbox
                        ok = rs.size <= 3 and hb[0] <= rs.bbox[0] and hb[1] <= rs.bbox[1] and hb[2] >= rs.bbox[2] and hb[3] >= rs.bbox[3]
                    if ok:
                        host = ns; break
                if host is not None:
                    # A THING IN A FRAME OF THE WALL'S COLOUR IS NOT THE WALL: what
                    # grows the terrain is its own colour only; the other colours
                    # inside (a block that rose with its black outline) are a thing
                    # of their own, read next
                    if mode == "body" and "terrain" in host.tags and len(rs.colors) > 1:
                        own = {p: col for p, col in rs.cells.items() if col == host.body}
                        rest = {p: col for p, col in rs.cells.items() if col != host.body}
                        for p, col in own.items():
                            host.cells[p] = col
                        events.append(Event("grew", host.sid, {"delta": len(own), "vec": (0.0, 0.0)}))
                        res_sprites.remove(rs)
                        for piece in read_frame(_paint(rest, frame.shape, bg), bg, glue_blocks=False):
                            res_sprites.append(piece)
                        if _DEBUG:
                            print(f"      residual {(rs.body, rs.size, rs.bbox)}: {len(own)} cells grow the terrain S{host.sid}; {len(rest)} of other colours are a thing of their own")
                        continue
                    if _DEBUG:
                        print(f"      residual {(rs.body, rs.size, rs.bbox)} grows S{host.sid} ({host.body}, {host.size}) as {mode}")
                    for p, col in rs.cells.items():
                        host.cells[p] = col
                    # WHAT GROWS INTO ITS NEIGHBOURS OF ITS OWN COLOUR IS ONE THING:
                    # small things of the host's colour now side by side with the
                    # grown host are absorbed (a mark's pixels, read apart, become
                    # the block that rose from them)
                    if mode == "body" and "terrain" not in host.tags and host.size <= 120:
                        hc = set(host.cells)
                        for sid2, ns2 in list(new_objects.items()):
                            if sid2 == host.sid or "terrain" in ns2.tags or "hud" in ns2.tags or ns2.body != host.body or len(ns2.colors) != 1 or ns2.size > 60:
                                continue
                            if any((r + dr, c + dc) in hc for (r, c) in ns2.cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                                for p, col in ns2.cells.items():
                                    host.cells[p] = col
                                del new_objects[sid2]
                                events.append(Event("vanished", sid2, {"into": host.sid}))
                                hc = set(host.cells)
                    # REVEALED, NOT GROWN: ground that shows where a mover
                    # stood was there all along — the previous exposure is
                    # amended so no transform is read into it
                    if "terrain" in host.tags and all(p in vacated for p in rs.cells):
                        for ps in self.prev:
                            if ps.sid == host.sid:
                                for p, col in rs.cells.items():
                                    if col == ps.body:
                                        ps.cells[p] = col
                        events.append(Event("revealed", host.sid, {"delta": rs.size}))
                    else:
                        events.append(Event("grew", host.sid, {"delta": rs.size, "vec": (0.0, 0.0)}))
                    res_sprites.remove(rs)
        # what is left is new: paired with what vanished nearby as a
        # change of form, else appeared
        vanished = [e for e in events if e.kind == "vanished"]
        # A BODY JUMPED: several things gone from one place and a residual
        # piece elsewhere that is their union (a two-colour player put
        # back at its start is read as one piece) — each keeps its name
        extra_res = []
        for rs in list(res_sprites):
            if "terrain" in rs.tags or rs.size >= BIG or len(rs.colors) < 2:
                continue
            olds = []
            for e in vanished:
                old = self.objects.get(e.sid) or self.hidden.get(e.sid)
                if old is not None and "terrain" not in old.tags and "hud" not in old.tags and old.body in rs.colors:
                    olds.append(old)
            if len(olds) < 2:
                continue
            # the group: olds touching one another, whose colours are rs's
            group = []
            for o in sorted(olds, key=lambda o: -o.size):
                if not group or any(max(o.bbox[0] - g.bbox[2], g.bbox[0] - o.bbox[2], o.bbox[1] - g.bbox[3], g.bbox[1] - o.bbox[3]) <= 1 for g in group):
                    group.append(o)
            if len(group) < 2 or abs(sum(o.size for o in group) - rs.size) > max(2, 0.2 * rs.size):
                continue
            gr0 = min(q[0] for o in group for q in o.cells); gc0 = min(q[1] for o in group for q in o.cells)
            dv = (rs.bbox[0] - gr0, rs.bbox[1] - gc0)
            hits = 0; total = 0
            for o in group:
                for (r, c), col in o.cells.items():
                    total += 1
                    if rs.cells.get((r + dv[0], c + dv[1])) == col:
                        hits += 1
            if hits < 0.9 * total:
                continue
            res_sprites.remove(rs)
            for o in group:
                e = next(e_ for e_ in vanished if e_.sid == o.sid)
                vanished.remove(e); events.remove(e)
                cells_o = {(r + dv[0], c + dv[1]): rs.cells.get((r + dv[0], c + dv[1]), col) for (r, c), col in o.cells.items()
                           if (r + dv[0], c + dv[1]) in rs.cells}
                ns = Sprite(sid=o.sid, cells=cells_o, parts=o.parts, tags=set(o.tags))
                new_objects[o.sid] = ns
                vec = (ns.centre[0] - o.centre[0], ns.centre[1] - o.centre[1])
                events.append(Event("moved", o.sid, {"vec": vec}))
                events.append(Event("jumped", o.sid, {"vec": vec}))
        # A THING JUMPED INTO A LARGER READING: a vanished thing whose exact
        # cells (shape and colours) lie inside a residual piece is that part
        # of the piece (the blue half of a player put back beside its
        # orange half, read as one) — split off, name kept
        for rs in list(res_sprites):
            if "terrain" in rs.tags or rs.size >= BIG:
                continue
            for e in list(vanished):
                old = self.objects.get(e.sid) or self.hidden.get(e.sid)
                if old is None or "terrain" in old.tags or "hud" in old.tags or old.size < 2 or old.size >= rs.size:
                    continue
                oc = list(old.cells.items())
                (r0, c0), col0 = oc[0]
                dv = None
                for (r, c), col in rs.cells.items():
                    if col != col0:
                        continue
                    d_ = (r - r0, c - c0)
                    if all(rs.cells.get((pr + d_[0], pc + d_[1])) == pcol for (pr, pc), pcol in oc):
                        dv = d_; break
                if dv is None:
                    continue
                vanished.remove(e); events.remove(e)
                cells_o = {(pr + dv[0], pc + dv[1]): pcol for (pr, pc), pcol in oc}
                ns = Sprite(sid=old.sid, cells=cells_o, parts=old.parts, tags=set(old.tags))
                new_objects[old.sid] = ns
                vec = (ns.centre[0] - old.centre[0], ns.centre[1] - old.centre[1])
                events.append(Event("moved", old.sid, {"vec": vec}))
                events.append(Event("jumped", old.sid, {"vec": vec}))
                rest = {p: col for p, col in rs.cells.items() if p not in cells_o}
                res_sprites.remove(rs)
                if rest:
                    for piece in read_frame(_paint(rest, frame.shape, bg), bg, glue_blocks=False):
                        res_sprites.append(piece)
                break
        for rs in res_sprites:
            if "terrain" in rs.tags or rs.size >= BIG:
                self._new(rs); new_objects[rs.sid] = rs
                events.append(Event("appeared", rs.sid, {"size": rs.size, "body": rs.body}))
                continue
            pair = None
            # A JUMP: the very thing gone from one place and standing in
            # another, however far — a teleport, a put-back after a death —
            # keeps its identity.  WHAT CHANGES UNDER A PRESS IS ONE WHOLE
            # THING (the starting theory): the same colour and shape first;
            # failing that the same shape in another colour (a thing that
            # moved and recoloured is still the thing), held until
            # contradicted
            for pass_ in ("colour", "shape"):
                for e in vanished:
                    old = self.objects.get(e.sid) or self.hidden.get(e.sid)
                    if old is None or "terrain" in old.tags or "hud" in old.tags or old.size < 2:
                        continue
                    if pass_ == "colour":
                        ok = old.body == rs.body and old.dims == rs.dims and abs(old.size - rs.size) <= max(2, 0.2 * old.size)
                    else:
                        ok = old.size >= 4 and old.shape == rs.shape and len(old.colors) == 1 and len(rs.colors) == 1
                    if ok:
                        d = abs(rs.centre[0] - old.centre[0]) + abs(rs.centre[1] - old.centre[1])
                        if pair is None or d < pair[0]:
                            pair = (d, e, "jumped")
                if pair is not None:
                    break
            if pair is not None:
                d, e, _ = pair; vanished.remove(e); events.remove(e)
                old = self.objects.get(e.sid) or self.hidden.get(e.sid)
                rs.sid = e.sid; rs.tags = set(old.tags); new_objects[rs.sid] = rs
                vec = (rs.centre[0] - old.centre[0], rs.centre[1] - old.centre[1])
                events.append(Event("moved", rs.sid, {"vec": vec}))
                events.append(Event("jumped", rs.sid, {"vec": vec}))
                if old.body != rs.body:
                    events.append(Event("recoloured", rs.sid, {"n": rs.size, "livery": sorted(set(rs.cells.values()))}))
                continue
            for e in vanished:
                old = self.objects.get(e.sid) or self.hidden.get(e.sid)
                if old is None or "terrain" in old.tags:
                    continue
                d = abs(rs.centre[0] - old.centre[0]) + abs(rs.centre[1] - old.centre[1])
                if d <= 24 and (pair is None or d < pair[0]):
                    pair = (d, e)
            if pair is not None:
                e = pair[1]; vanished.remove(e); events.remove(e)
                old = self.objects.get(e.sid) or self.hidden.get(e.sid)
                rs.sid = e.sid; new_objects[rs.sid] = rs
                vec = (rs.centre[0] - old.centre[0], rs.centre[1] - old.centre[1])
                events.append(Event("morphed", rs.sid, {"vec": vec, "delta": rs.size - old.size, "body": rs.body, "was": old.body}))
                if abs(vec[0]) + abs(vec[1]) > 0.5:
                    events.append(Event("moved", rs.sid, {"vec": vec}))
            else:
                # A THING GONE TO NOTHING IS STILL A THING: what grows again of
                # its colour where it stood takes its name (vc33: a water column
                # pumped away one press and refilled the next kept no laws as a
                # new thing each time)
                gh_ = None
                for gsid_, (gs_, age_) in (list(self.ghosts.items()) if not _ENV.get("NO_GHOSTS") else []):
                    if gs_.body != rs.body or gsid_ in new_objects:
                        continue
                    gb_ = gs_.bbox
                    if all(gb_[0] - 1 <= r <= gb_[2] + 1 and gb_[1] - 1 <= c <= gb_[3] + 1 for r, c in rs.cells) and (gb_[2] == rs.bbox[2] or gb_[0] == rs.bbox[0] or gb_[1] == rs.bbox[1] or gb_[3] == rs.bbox[3]):
                        gh_ = gsid_; break
                if gh_ is not None:
                    rs.sid = gh_; rs.tags = set(self.ghosts[gh_][0].tags) - {"hidden"}; new_objects[rs.sid] = rs
                    self.ghosts.pop(gh_, None)
                    events.append(Event("appeared", rs.sid, {"size": rs.size, "body": rs.body, "regrown": True}))
                else:
                    self._new(rs); new_objects[rs.sid] = rs
                    events.append(Event("appeared", rs.sid, {"size": rs.size, "body": rs.body}))
        # ABSORBED: a thing whose every cell lies inside a larger thing of
        # its own body colour is that thing now (a same-coloured piece the
        # mover stood on and "carried" — one colour is one thing there)
        for sid in list(new_objects):
            ns = new_objects[sid]
            if sid >= 1000 or "terrain" in ns.tags or "hud" in ns.tags or ns.size > 16:
                continue
            for sid2, big in new_objects.items():
                if sid2 == sid or sid2 >= 1000 or big.size <= ns.size or big.body != ns.body or "terrain" in big.tags:
                    continue
                if all(p in big.cells for p in ns.cells):
                    del new_objects[sid]
                    events = [e for e in events if e.sid != sid]
                    events.append(Event("vanished", sid, {"size": ns.size, "body": ns.body, "into": sid2}))
                    break
        # tags follow the current reading (a thing that became large is
        # terrain; hud strips)
        for sid, ns in new_objects.items():
            r0, c0, r1, c1 = ns.bbox
            if ns.dims[0] <= 3 and (r0 == 0 or r1 == H - 1):
                ns.tags.add("hud")
            thin_wide = (ns.dims[0] <= 3 and ns.dims[1] >= 12) or (ns.dims[1] <= 3 and ns.dims[0] >= 12)
            at_edge = r0 <= 1 or r1 >= H - 2 or c0 <= 1 or c1 >= W - 2
            if thin_wide and at_edge and ns.is_bar():
                ns.tags.add("bar")
        for strip in hud_strips(list(new_objects.values()), H, W):
            for s in strip:
                s.tags.add("hud")
        # small LIKE things in a row in the bar's band (lives beside a step
        # bar) are hud too — two or more of one shape, never a lone piece
        bars = [ns for ns in new_objects.values() if "bar" in ns.tags]
        in_band = []
        for ns in new_objects.values():
            if ns.size <= 9 and "bar" not in ns.tags:
                b = ns.bbox
                for bar in bars:
                    bb = bar.bbox
                    if (bar.dims[0] <= 3 and abs(b[0] - bb[0]) <= 2 and abs(b[2] - bb[2]) <= 2) or \
                            (bar.dims[1] <= 3 and abs(b[1] - bb[1]) <= 2 and abs(b[3] - bb[3]) <= 2):
                        in_band.append(ns); break
        shapes = Counter((ns.body, ns.shape) for ns in in_band)
        for ns in in_band:
            if shapes[(ns.body, ns.shape)] >= 2:
                ns.tags.add("hud")
        for ns in new_objects.values():
            if ns.body in getattr(self, "ground", set()):
                ns.tags.add("terrain")
        # THE CAMERA: the one shift that carries the last frame onto this
        # one (found before the things were placed) is a SCROLL — the world
        # did not move, the screen did.  Reported; whoever keeps places
        # remembers them shifted
        self.scroll = None
        if getattr(self, "scroll_hint", None) is not None:
            u = self.scroll_hint
            self.scroll = u
            self.camera = (getattr(self, "camera", (0, 0))[0] - u[0], getattr(self, "camera", (0, 0))[1] - u[1])
            events.append(Event("scrolled", -1, {"vec": (float(u[0]), float(u[1]))}))
        # A THING IS ONE PIECE: an object whose cells fall into two
        # 4-connected pieces after a press was two things read as one — two
        # like blocks touching, one of them pushed away (tw01).  The piece
        # holding most of the thing keeps its identity; each other piece is
        # a thing of its own from now on.  (Terrain and HUD are left alone;
        # a thing of several colours — a body with its marks — too.)
        for sid_ in (list(new_objects.keys()) if not _ENV.get("NO_SPLIT") else []):
            o_ = new_objects[sid_]
            if "terrain" in o_.tags or "hud" in o_.tags or len(set(o_.cells.values())) != 1 or len(o_.cells) < 4:
                continue
            pieces_ = _four_pieces(o_.cells)
            if len(pieces_) != 2:
                continue
            # (only a split BY A MOVE: one piece is part of the thing as it stood,
            # the other that remainder carried whole by one vector — not a canvas
            # painted into two regions, cd82)
            old_ = self.objects.get(sid_)
            if old_ is None or len(old_.cells) != len(o_.cells):
                continue
            oldc_ = set(old_.cells)
            ok_split_ = False
            for keep_, moved_ in ((pieces_[0], pieces_[1]), (pieces_[1], pieces_[0])):
                if not keep_ <= oldc_:
                    continue
                rest_ = oldc_ - keep_
                if len(rest_) != len(moved_):
                    continue
                r0_, c0_ = min(rest_); r1_, c1_ = min(moved_)
                v_ = (r1_ - r0_, c1_ - c0_)
                if v_ != (0, 0) and {(r + v_[0], c + v_[1]) for (r, c) in rest_} == moved_:
                    ok_split_ = True
                    pieces_ = [keep_, moved_]
                    break
            if not ok_split_:
                continue
            if _DEBUG or _ENV.get("SPLIT_DEBUG"):
                print(f"      S{sid_} ({len(o_.cells)} cells, colour {next(iter(set(o_.cells.values())))}) fell into two pieces by a move: {len(pieces_[0])} kept at {min(pieces_[0])}, {len(pieces_[1])} moved to {min(pieces_[1])}")
            o_.cells = {p_: o_.cells[p_] for p_ in pieces_[0]}
            for pc_ in pieces_[1:]:
                nw_ = self._new(Sprite(sid=-1, cells={p_: o_.cells.get(p_, next(iter(set(o_.cells.values())))) for p_ in pc_}, parts=1))
                nw_.cells = {p_: next(iter(set(o_.cells.values()))) for p_ in pc_}
                nw_.tags = set(o_.tags)
                nw_.split_from = sid_; nw_.split_vec = (int(v_[0]), int(v_[1])); nw_.split_at = getattr(self, "n_updates", 0)   # (the law-maker reads it as a thing pushed)
                o_.split_lost = (int(v_[0]), int(v_[1])); o_.split_at = getattr(self, "n_updates", 0)                         # (the kept piece lost nothing: the other went)
                new_objects[nw_.sid] = nw_
                events.append(Event("born", nw_.sid, {"split_from": sid_, "vec": (float(v_[0]), float(v_[1]))}))
        self._learn_units(new_objects, events)
        self._split_units(new_objects, events)
        # (the ghosts: what vanished this press without being named again,
        # kept at its place for a while — forty presses, hand-chosen)
        for e_ in events:
            if e_.kind == "vanished" and e_.sid not in new_objects and e_.sid not in new_hidden:
                old_ = self.objects.get(e_.sid) or self.hidden.get(e_.sid)
                if old_ is not None and "hud" not in old_.tags and old_.size >= 2:
                    self.ghosts[e_.sid] = (old_, 0)
        for gsid_ in list(self.ghosts):
            gs_, age_ = self.ghosts[gsid_]
            if gsid_ in new_objects or age_ >= 40:
                self.ghosts.pop(gsid_, None)
            else:
                self.ghosts[gsid_] = (gs_, age_ + 1)
        self.objects = new_objects
        self.hidden = new_hidden
        self.prev = self._expose()
        self.last_events = events
        return events

    def rename(self, b, a, vec=None):
        """Give object `b` the identity `a` — an id that has just vanished
        (whoever knows better says the thing at b IS the thing a, arrived:
        the merge the frame showed as a recolouring).  The events read as
        a's move."""
        ob = self.objects.get(b)
        if ob is None or a in self.objects or a in self.hidden:
            return False
        del self.objects[b]
        ob.sid = a
        self.objects[a] = ob
        self.prev = self._expose()
        self.last_events = [e for e in self.last_events if not ((e.kind == "vanished" and e.sid == a) or e.sid == b)]
        self.last_events.append(Event("moved", a, {"vec": (float(vec[0]), float(vec[1])) if vec is not None else (0.0, 0.0)}))
        return True

    def swap_ids(self, a, b):
        """Give object `b` the identity `a` and vice versa (a reading that
        put a name on the wrong thing, corrected by whoever knows better —
        the body that moves together stays together)."""
        oa = self.objects.get(a) or self.hidden.get(a)
        ob = self.objects.get(b) or self.hidden.get(b)
        if oa is None or ob is None:
            return False
        for store in (self.objects, self.hidden):
            ia = a in store; ib = b in store
            if ia:
                del store[a]
            if ib:
                del store[b]
            if ia:
                store[b] = oa
            if ib:
                store[a] = ob
        oa.sid, ob.sid = b, a
        self.prev = self._expose()
        for e in self.last_events:
            if e.sid == a:
                e.sid = b
            elif e.sid == b:
                e.sid = a
        return True

    # ------------------------------------------------------------ exposure
    def _learn_units(self, new_objects, events):
        """THE UNIT OF A KIND: the smallest whole rectangle of one colour ever
        seen to MOVE alone is the kind's unit (carried across levels).  A
        colour whose things are ELASTIC — seen to grow or shrink (a piston,
        a bar) — has no units: a long bar is one thing, not a row of them
        (s5i5 L2's pistons, split into pieces, lost the level)."""
        uf_ = getattr(self, "unit_form", None)
        if uf_ is None:
            uf_ = self.unit_form = {}
        el_ = getattr(self, "elastic", None)
        if el_ is None:
            el_ = self.elastic = set()
        # (a thing that grew by a like thing joining it, or shrank by one
        # leaving — a vanish or a birth of its colour by that very amount
        # this press — did not stretch: two blocks met or parted)
        joined_ = defaultdict(set)
        for e_ in events:
            if e_.kind in ("vanished", "born", "covered", "uncovered", "hidden"):
                sz_ = e_.data.get("size")
                col_ = e_.data.get("body")
                o2_ = new_objects.get(e_.sid) or self.objects.get(e_.sid) or self.hidden.get(e_.sid)
                if o2_ is not None:
                    sz_ = len(o2_.cells) if sz_ is None else sz_
                    col_ = o2_.body if col_ is None else col_
                if sz_ is not None and col_ is not None:
                    joined_[col_].add(int(sz_))
        for sid_, o_ in new_objects.items():
            old_ = self.objects.get(sid_)
            if old_ is None or "terrain" in o_.tags or "hud" in o_.tags:
                continue
            cols_ = set(o_.cells.values())
            if len(cols_) == 1 and abs(len(old_.cells) - len(o_.cells)) in joined_.get(next(iter(cols_)), ()):
                continue
            if len(cols_) == 1 and len(set(old_.cells.values())) == 1 and len(old_.cells) != len(o_.cells) and getattr(o_, "split_at", None) != getattr(self, "n_updates", -1):
                if _DEBUG or _ENV.get("SPLIT_DEBUG"):
                    print(f"      S{sid_} colour {next(iter(cols_))} changed size {len(old_.cells)} -> {len(o_.cells)}: elastic (events {[(e_.kind, e_.sid, e_.data.get('size')) for e_ in events if e_.kind not in ('moved',)]}; joined {dict(joined_)})")
                el_.add(next(iter(cols_)))
        # A COLOUR A THING PUTS ON AND OFF IS A LIVERY, NOT A KIND: a thing
        # recoloured whole, where it stands, from one colour to another (a
        # bar chosen by a click wears the chosen colour; the bar let go wears
        # the plain one) — neither colour has a unit: the unit of a kind is
        # of the thing, not of its coat (sp80 L3: the 9-coat's unit, learned
        # on a 3-unit bar in L2, split the 6-unit bar into two things, and
        # every press of the level was read as a birth and a death)
        if not _ENV.get("NO_LIVERY_ELASTIC"):
            for e_ in events:
                if e_.kind != "recoloured":
                    continue
                o_ = new_objects.get(e_.sid)
                old_ = self.objects.get(e_.sid)
                if o_ is None or old_ is None or "terrain" in o_.tags or "hud" in o_.tags or len(o_.cells) < 4:
                    continue
                ca_, cb_ = set(old_.cells.values()), set(o_.cells.values())
                if len(ca_) == 1 and len(cb_) == 1 and ca_ != cb_ and set(old_.cells.keys()) == set(o_.cells.keys()):
                    for c_ in ca_ | cb_:
                        if c_ not in el_ and (_DEBUG or _ENV.get("SPLIT_DEBUG")):
                            print(f"      colour {c_} is a livery of S{e_.sid} ({next(iter(ca_))} -> {next(iter(cb_))}): no unit")
                        el_.add(c_)
        for c_ in el_:
            uf_.pop(c_, None)
        for e_ in events:
            if e_.kind != "moved" or e_.sid not in new_objects:
                continue
            o_ = new_objects[e_.sid]
            cols_ = set(o_.cells.values())
            if len(cols_) != 1 or "terrain" in o_.tags or "hud" in o_.tags or len(o_.cells) < 4:
                continue
            # (moved WHOLE — shifted by one vector with every cell kept: a
            # region's remnant that "moved" when a thing covered part of it
            # is no unit — wa30's pen fill, read as 4×2 units, lost L2)
            old_ = self.objects.get(e_.sid)
            if getattr(o_, "moved_by", None) is None or old_ is None or len(old_.cells) != len(o_.cells):
                continue
            b_ = o_.bbox; h_, w_ = b_[2] - b_[0] + 1, b_[3] - b_[1] + 1
            if h_ * w_ != len(o_.cells):
                continue
            col_ = next(iter(cols_))
            if col_ in el_:
                continue
            old_u = uf_.get(col_)
            if old_u is None or h_ * w_ < old_u[0] * old_u[1]:
                uf_[col_] = (h_, w_)

    def _split_units(self, new_objects, events):
        """A thing of a colour that is a whole row or column of the colour's
        units is as many things as units — two like blocks touching, read as
        one, pushed one at a time (tw01: the pair read as one thing moved by
        a push that moved half of it, and the plan was lost in the halves)."""
        uf_ = getattr(self, "unit_form", None) or {}
        el_ = getattr(self, "elastic", None) or set()
        if not uf_ or _ENV.get("NO_UNITS"):
            return
        for sid_ in list(new_objects.keys()):
            o_ = new_objects[sid_]
            cols_ = set(o_.cells.values())
            if len(cols_) != 1 or "terrain" in o_.tags or "hud" in o_.tags or next(iter(cols_)) in el_:
                continue
            u_ = uf_.get(next(iter(cols_)))
            if u_ is None:
                continue
            b_ = o_.bbox; h_, w_ = b_[2] - b_[0] + 1, b_[3] - b_[1] + 1
            if h_ * w_ != len(o_.cells):
                continue
            if h_ == u_[0] and w_ % u_[1] == 0 and w_ // u_[1] >= 2:
                boxes_ = [(b_[0], b_[1] + i * u_[1], b_[2], b_[1] + (i + 1) * u_[1] - 1) for i in range(w_ // u_[1])]
            elif w_ == u_[1] and h_ % u_[0] == 0 and h_ // u_[0] >= 2:
                boxes_ = [(b_[0] + i * u_[0], b_[1], b_[0] + (i + 1) * u_[0] - 1, b_[3]) for i in range(h_ // u_[0])]
            else:
                continue
            col_ = next(iter(cols_))
            if _DEBUG or _ENV.get("SPLIT_DEBUG"):
                print(f"      S{sid_} ({len(o_.cells)} cells, colour {col_}) is {len(boxes_)} units of {u_}: split into units")
            o_.cells = {p_: col_ for p_ in o_.cells if boxes_[0][0] <= p_[0] <= boxes_[0][2] and boxes_[0][1] <= p_[1] <= boxes_[0][3]}
            for bx_ in boxes_[1:]:
                cells_ = {(r, c): col_ for r in range(bx_[0], bx_[2] + 1) for c in range(bx_[1], bx_[3] + 1)}
                nw_ = self._new(Sprite(sid=-1, cells=dict(cells_), parts=1))
                nw_.cells = cells_
                nw_.tags = set(o_.tags)
                new_objects[nw_.sid] = nw_
                events.append(Event("born", nw_.sid, {"unit_of": sid_}))

    def _expose(self):
        """Objects as body sprites, plus each non-body colour's cells as a
        PART pseudo-sprite whose id is anchored to the object."""
        out = []
        for sid, obj in list(self.objects.items()) + list(self.hidden.items()):
            hidden = sid in self.hidden
            body = obj.body
            bcells = {p: c for p, c in obj.cells.items() if c == body}
            parts = defaultdict(dict)
            for p, c in obj.cells.items():
                if c != body:
                    parts[c][p] = c
            if bcells:
                # the object's WHOLE footprint (a ring with its facing dot is
                # a solid square that translates); its parts are exposed
                # again below, so a mark or a tip keeps its own name
                bs = Sprite(sid=sid, cells=dict(obj.cells), parts=1, tags=set(obj.tags) | ({"hidden"} if hidden else set()))
                bs.moved_by = getattr(obj, "moved_by", None)
                # (a split by a move, this press: the kept piece lost nothing,
                # the piece that went was pushed — read once by the law-maker)
                if getattr(obj, "split_at", None) == getattr(self, "n_updates", -1):
                    bs.split_lost = getattr(obj, "split_lost", None)
                    bs.split_from = getattr(obj, "split_from", None)
                out.append(bs)
            for col, pc in parts.items():
                ps = Sprite(sid=sid * 1000 + col, cells=pc, parts=1, tags=set(t for t in obj.tags if t != "terrain") | ({"hidden"} if hidden else set()))
                ps.moved_by = getattr(obj, "moved_by", None)
                out.append(ps)
        return out

    def get(self, sid):
        for s in self.prev:
            if s.sid == sid:
                return s
        return None
