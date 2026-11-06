"""ENTITIES: the object-level substrate.

A frame is read into SPRITES — rigid, possibly multi-colour things — not
same-colour components.  A component that lies inside another's box (a
dot in a tip, a livery pixel in a block, the centre of a cross, the
brackets in a slider panel) is a PART of it; small touching components
that form a compact block (a card of three colours, a piece with its
ports) are one sprite.  Big components (walls, floors, panels) stay
alone.  These are PRIORS: the transition record corrects them — parts
that move apart are split for good, sprites that always move together
are one rigid body.

Every sprite has: colours (multiset), body colour, shape (normalised
cells), size, box, centre, parts, livery (cells whose colour differs
from the body's), cavity (enclosed empty cells), class key (shape +
colours), and an identity that survives motion, rotation, livery change
and merging with a neighbour.

Between two frames the tracker emits EVENTS in a closed vocabulary —
moved(id, vec), rotated(id, k), recoloured(id, cells), grew/shrank(id,
delta), appeared(id), vanished(id), merged(ids -> id), split(id -> ids)
— which is what the rule inducer fits templates to.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import os
import numpy as np

from ..perception import segment

BIG = 220            # a component this large is terrain, never a part
GLUE_BOX = 14        # touching parts glue only into a box this small
DENSE = 0.55         # ... that they fill at least this much


def _rebase(cells):
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def dihedral(norm):
    """(k, flip) -> image, for the eight symmetries of the plane."""
    out = {}
    cur = norm
    for k in range(4):
        out[(k, False)] = cur
        out[(k, True)] = _rebase([(r, -c) for r, c in cur])
        cur = _rebase([(c, -r) for r, c in cur])
    return out


@dataclass
class Sprite:
    sid: int                                 # identity (stable across frames)
    cells: dict                              # {(r, c): color}
    parts: int = 1
    tags: set = field(default_factory=set)   # 'hud', 'border', 'terrain'

    # ---- geometry
    @property
    def size(self):
        return len(self.cells)

    @property
    def bbox(self):
        rs = [r for r, _ in self.cells]; cs = [c for _, c in self.cells]
        return (min(rs), min(cs), max(rs), max(cs))

    @property
    def dims(self):
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0 + 1, c1 - c0 + 1)

    @property
    def centre(self):
        r0, c0, r1, c1 = self.bbox
        return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

    @property
    def colors(self):
        return Counter(self.cells.values())

    @property
    def body(self):
        """The dominant colour."""
        return self.colors.most_common(1)[0][0]

    @property
    def shape(self):
        return _rebase(self.cells.keys())

    @property
    def cshape(self):
        """Shape with colours, normalised: the class key's fine half."""
        r0, c0 = self.bbox[0], self.bbox[1]
        return frozenset(((r - r0, c - c0), col) for (r, c), col in self.cells.items())

    @property
    def livery(self):
        b = self.body
        return frozenset(p for p, col in self.cells.items() if col != b)

    @property
    def key(self):
        """Class key: same shape, same colour multiset (livery aside)."""
        return (self.shape, tuple(sorted(self.colors.items())))

    @property
    def body_key(self):
        """Coarser class: shape and body colour only (livery-blind)."""
        return (self.shape, self.body)

    def cavity(self):
        r0, c0, r1, c1 = self.bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            return frozenset()
        free = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in self.cells}
        border = [p for p in free if p[0] in (r0, r1) or p[1] in (c0, c1)]
        seen = set(border); stack = list(border)
        while stack:
            r, c = stack.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in free and q not in seen:
                    seen.add(q); stack.append(q)
        return frozenset(free - seen)

    def is_bar(self):
        h, w = self.dims
        return self.size == h * w and max(h, w) >= 3 * min(h, w)

    def __repr__(self):
        cols = "/".join(f"{c}x{n}" for c, n in self.colors.most_common(3))
        return f"S{self.sid}[{cols} {self.dims[0]}x{self.dims[1]}@{self.bbox[:2]}]"


# ------------------------------------------------------------ one frame

def _touch(a, b):
    """8-adjacent bboxes (gap 0)."""
    ar0, ac0, ar1, ac1 = a; br0, bc0, br1, bc1 = b
    return not (br0 > ar1 + 1 or ar0 > br1 + 1 or bc0 > ac1 + 1 or ac0 > bc1 + 1)


def _adjacent(cells_a, cells_b):
    """Some cell of a has an 8-neighbour in b (a part touches its whole;
    a dot merely lying within a cross's box does not)."""
    for r, c in cells_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in cells_b:
                    return True
    return False


def _inside(a, b):
    """bbox a within bbox b."""
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def tag_textures(sprites, logical=None, bg=None):
    if os.environ.get("NO_TEXTURE"):
        return sprites
    """A KIND IS A TEXTURE, NOT A COLOUR: a thing whose cells, within its
    box, repeat with a period along its long side (three repeats or more,
    with holes showing the floor between) is a MATERIAL — hatching, brick,
    water — one thing of its own kind, whatever colour its strokes are.
    Its kind differs from a solid fill of the same colour (fiddle.kind).
    The hand-chosen bounds: density 0.3–0.9 of its box, period 2–8,
    at least three repeats, at least 3 cells wide."""
    for s in sprites:
        if "hud" not in s.tags and _periodic(s):
            s.tags.add("texture")
    # PIECES THAT TOGETHER REPEAT ARE ONE MATERIAL: same-colour pieces
    # standing within a cell of each other (strokes of a hatching, bricks
    # with mortar between) are tried as one thing; when that union's cells
    # repeat with a period along its long side, it is one textured thing
    cand = [s for s in sprites if "hud" not in s.tags and "texture" not in s.tags and s.size >= 3 and len(set(s.cells.values())) == 1]
    parent = {id(s): id(s) for s in cand}
    by_id = {id(s): s for s in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            sa, sb = cand[a], cand[b]
            if sa.body != sb.body:
                continue
            A, B = sa.bbox, sb.bbox
            if max(A[0] - B[2], B[0] - A[2], A[1] - B[3], B[1] - A[3]) <= 1:
                parent[find(id(sa))] = find(id(sb))
    groups = defaultdict(list)
    for s in cand:
        groups[find(id(s))].append(s)
    merged = set(); extra = []
    for members in groups.values():
        if len(members) < 2:
            continue
        cells = {}
        for s in members:
            cells.update(s.cells)
        u = Sprite(sid=-1, cells=cells, parts=len(members), tags=set())
        if _periodic(u):
            for s in members:
                merged.add(id(s)); u.tags |= s.tags
            u.tags.add("texture")
            extra.append(u)
    if merged:
        sprites = [s for s in sprites if id(s) not in merged] + extra
    # THE PATTERN GOES ON UNDER WHAT COVERS IT: two textured things of one
    # colour, aligned, with a gap between them no longer than two periods
    # whose cells all show something else (a crate on the hatching) are
    # one material — its box spans the gap; the thing in the gap stands
    # ON it (a kind seen on a material passes it)
    if logical is not None:
        tex = [s for s in sprites if "texture" in s.tags]
        joined = True
        while joined and len(tex) >= 2:
            joined = False
            for a in range(len(tex)):
                for b in range(a + 1, len(tex)):
                    A, B = tex[a], tex[b]
                    if A.body != B.body:
                        continue
                    ab, bb = A.bbox, B.bbox
                    for axis in (0, 1):
                        o0, o1 = (1, 3) if axis == 0 else (0, 2)
                        if ab[o0] != bb[o0] or ab[o1] != bb[o1]:
                            continue            # (not aligned across)
                        lo, hi = (A, B) if ab[axis] <= bb[axis] else (B, A)
                        gap0, gap1 = lo.bbox[axis + 2] + 1, hi.bbox[axis] - 1
                        if gap1 < gap0 or gap1 - gap0 + 1 > 16:
                            continue
                        cells_gap = [(r, c) for r in (range(gap0, gap1 + 1) if axis == 0 else range(ab[0], ab[2] + 1))
                                     for c in (range(ab[1], ab[3] + 1) if axis == 0 else range(gap0, gap1 + 1))]
                        if all(int(logical[r, c]) != bg and int(logical[r, c]) != A.body for r, c in cells_gap):
                            cells = dict(A.cells); cells.update(B.cells)
                            # (the strokes go on under the cover: the gap is
                            # filled by the period — the planner meets the
                            # material there as it will when the cover moves)
                            per = _period_of(lo)
                            if per:
                                lo_cells = set(lo.cells)
                                for (r, c) in cells_gap:
                                    q = (r, c)
                                    while q[0] > lo.bbox[2] if axis == 0 else q[1] > lo.bbox[3]:
                                        q = (q[0] - per, q[1]) if axis == 0 else (q[0], q[1] - per)
                                    if q in lo_cells:
                                        cells[(r, c)] = A.body
                            u = Sprite(sid=-1, cells=cells, parts=A.parts + B.parts, tags=(A.tags | B.tags))
                            u.hull = (min(ab[0], bb[0]), min(ab[1], bb[1]), max(ab[2], bb[2]), max(ab[3], bb[3]))
                            sprites = [s for s in sprites if s is not A and s is not B] + [u]
                            tex = [s for s in sprites if "texture" in s.tags]
                            joined = True
                            break
                    if joined:
                        break
                if joined:
                    break
    return sprites


def _period_of(s):
    """The period along the long side of a textured thing, or 0."""
    r0, c0, r1, c1 = s.bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    pres = set(s.cells)
    axis = 0 if h >= w else 1
    n = h if axis == 0 else w
    for p in range(2, 9):
        if n < 2 * p:
            break
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                q = (r + p, c) if axis == 0 else (r, c + p)
                if q[0] > r1 or q[1] > c1:
                    continue
                if ((r, c) in pres) != (q in pres):
                    ok = False; break
            if not ok:
                break
        if ok:
            return p
    return 0


def _periodic_cells(pres, bbox):
    """The cells within `bbox` repeat along its long side with a period of
    2–8, three repeats or more, holes between (density 0.3–0.9), 3+ wide."""
    r0, c0, r1, c1 = bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if min(h, w) < 3 or h * w < 32:
        return False
    n_in = sum(1 for (r, c) in pres if r0 <= r <= r1 and c0 <= c <= c1)
    d = n_in / float(h * w)
    if not (0.3 <= d <= 0.9):
        return False
    axis = 0 if h >= w else 1
    n = h if axis == 0 else w
    for p in range(2, 9):
        if n < 3 * p:
            break
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                q = (r + p, c) if axis == 0 else (r, c + p)
                if q[0] > r1 or q[1] > c1:
                    continue
                if ((r, c) in pres) != (q in pres):
                    ok = False; break
            if not ok:
                break
        if ok:
            return True
    return False


def _periodic(s):
    """A texture: the cells of `s` repeat along its long side (see
    _periodic_cells) — or `s` is a RING of one: its cells all within a band
    of eight or fewer along its box's border, nothing of it inside, and
    each side's strip periodic along its length (a hatched fence round a
    yard — wa30 L4: the ring failed the straight test and was a solid fill
    of its colour to every organ)."""
    pres = set(s.cells)
    if _periodic_cells(pres, s.bbox):
        return True
    if os.environ.get("NO_RING_TEXTURE"):
        return False
    r0, c0, r1, c1 = s.bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if min(h, w) < 12:
        return False
    for b in range(2, 9):
        if 2 * b >= min(h, w) - 2:
            break
        inner = (r0 + b, c0 + b, r1 - b, c1 - b)
        if any(inner[0] <= r <= inner[2] and inner[1] <= c <= inner[3] for (r, c) in pres):
            continue
        strips = [(r0, c0, r0 + b - 1, c1), (r1 - b + 1, c0, r1, c1), (r0, c0, r1, c0 + b - 1), (r0, c1 - b + 1, r1, c1)]
        if all(_periodic_cells(pres, st) for st in strips):
            return True
        break
    return False


def read_frame(logical, bg, split_keys=(), hud_rows=(), next_id=(0,), glue_blocks=True):
    """Components -> sprites by the static priors.  `split_keys`: pairs
    of body-colour keys the record has shown to move apart (never glued
    again).  Returns a list of Sprite with provisional ids (the tracker
    assigns identity)."""
    objs = segment(logical, bg, mask=None)
    n = len(objs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)
    small = [o.size < BIG for o in objs]
    # GROUND: a colour that covers a large share of the frame is floor or
    # wall; its pieces are never parts of a sprite (ka59: a floor patch
    # wedged between a block and the pit is not part of the block)
    area = Counter(); biggest = Counter()
    for o in objs:
        area[o.color] += o.size
        biggest[o.color] = max(biggest[o.color], o.size)
    ground = {c for c, a in area.items() if a >= 300 and biggest[c] >= 100}

    def line(o):
        h, w = o.bbox[2] - o.bbox[0] + 1, o.bbox[3] - o.bbox[1] + 1
        return min(h, w) <= 1 and max(h, w) >= 6

    def crossing(a, b):
        return not (b[0] > a[2] or a[0] > b[2] or b[1] > a[3] or a[1] > b[3])
    # a thin line (a rope, a connector, a grid line) glues only to a line
    # it CROSSES (the two bars of a cross), never to what it touches
    glue = [small[i] and objs[i].color not in ground for i in range(n)]
    # A MATERIAL IS READ BEFORE THE THINGS ON IT: same-colour pieces within
    # a cell of one another (the strokes of a hatching) tried as one thing
    # FIRST — when their union repeats (a strip, or a ring of strips), they
    # are one material, and no piece of another colour standing against
    # them is glued to them (a block against a hatched fence is a block;
    # wa30 L4's fence was read in six pieces, two of them with blocks in).
    # (WITHDRAWN TO A SWITCH, 2026-10-25: right about the fence, wrong for
    # m0r0 L2 — its checker, read whole as one material instead of lone
    # cells the body was glued to, undid the death reading the plan lived
    # by; the hazard organ does not yet read materials.  MATERIAL_FIRST=1)
    material = set()
    if os.environ.get("MATERIAL_FIRST"):
        try:
            mpar = list(range(n))

            def mfind(i):
                while mpar[i] != i:
                    mpar[i] = mpar[mpar[i]]; i = mpar[i]
                return i
            by_col = defaultdict(list)
            for i in range(n):
                if glue[i]:
                    by_col[objs[i].color].append(i)
            for col, idx in by_col.items():
                if len(idx) < 3:
                    continue
                for a in range(len(idx)):
                    A = objs[idx[a]].bbox
                    for b in range(a + 1, len(idx)):
                        B = objs[idx[b]].bbox
                        if max(A[0] - B[2], B[0] - A[2], A[1] - B[3], B[1] - A[3]) <= 1:
                            mpar[mfind(idx[a])] = mfind(idx[b])
            mgroups = defaultdict(list)
            for i in range(n):
                if glue[i]:
                    mgroups[mfind(i)].append(i)
            for members in mgroups.values():
                if len(members) < 3:
                    continue
                cells = {}
                for i in members:
                    cells.update({q: objs[i].color for q in objs[i].cells})
                if _periodic(Sprite(sid=-1, cells=cells, parts=len(members), tags=set())):
                    material |= set(members)
                    for i in members[1:]:
                        union(members[0], i)
        except Exception:
            material = set()
    # rule A: a component inside another's box is a part of it
    for i in range(n):
        for j in range(n):
            if i == j or not small[j] or not small[i] or objs[i].color in ground or objs[j].color in ground:
                continue        # a picture on the ground (or in a HUD box) is its own thing
            if (i in material) != (j in material):
                continue        # (a material and a thing of another colour are not one)
            if objs[i].size <= objs[j].size and _inside(objs[i].bbox, objs[j].bbox) \
                    and (objs[i].color, objs[j].color) not in split_keys \
                    and _adjacent(objs[i].cells, objs[j].cells):
                union(i, j)
    # rule B: touching small components forming a compact, dense block
    for i in range(n):
        if not glue[i] or not glue_blocks:
            continue
        for j in range(i + 1, n):
            if not glue[j] or find(i) == find(j):
                continue
            if (objs[i].color, objs[j].color) in split_keys or (objs[j].color, objs[i].color) in split_keys:
                continue
            if (i in material) != (j in material):
                continue        # (a material and a thing of another colour are not one)
            if not _touch(objs[i].bbox, objs[j].bbox):
                continue
            a, b = objs[i].bbox, objs[j].bbox
            if line(objs[i]) or line(objs[j]):
                if line(objs[i]) and line(objs[j]) and crossing(a, b) and objs[i].color == objs[j].color:
                    union(i, j)
                continue
            ub = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            h, w = ub[2] - ub[0] + 1, ub[3] - ub[1] + 1
            if h <= GLUE_BOX and w <= GLUE_BOX and (objs[i].size + objs[j].size) >= DENSE * h * w:
                union(i, j)
            # rule C: tiny same-colour pieces touching corner to corner
            # are one drawing (a diamond outline of four dots)
            elif objs[i].color == objs[j].color and objs[i].size <= 6 and objs[j].size <= 6 \
                    and h <= 8 and w <= 8:
                union(i, j)
    # rule D: A LINE BROKEN BY A DOT IS ONE LINE — a tiny component (a cell
    # or two) with the same colour on both sides of it, above and below or
    # left and right, is a part of the thing it interrupts (the black
    # centre of a cross), and the pieces on either side are one thing
    cell_owner = {}
    for i, o in enumerate(objs):
        for p in o.cells:
            cell_owner[p] = i
    for i, o in enumerate(objs):
        if o.size > 2 or o.color in ground:
            continue
        for (r, c) in o.cells:
            for (p1, p2) in (((r - 1, c), (r + 1, c)), ((r, c - 1), (r, c + 1))):
                a, b = cell_owner.get(p1), cell_owner.get(p2)
                if a is None or b is None or a == i or b == i:
                    continue
                if objs[a].color != objs[b].color or objs[a].color == o.color or objs[a].color in ground:
                    continue
                if (o.color, objs[a].color) in split_keys or (objs[a].color, o.color) in split_keys:
                    continue
                union(i, a); union(i, b)
    # rule E: A DOTTED LINE IS ONE LINE — tiny pieces of one colour that
    # touch corner to corner (a diagonal drawn cell by cell) are one drawing
    # (only pieces that nothing larger has claimed: a line's end pixel
    # lying in a thing's box belongs to that thing, and must not drag the
    # whole line and what it touches into one)
    gsize = Counter()
    for i in range(n):
        gsize[find(i)] += objs[i].size

    def diag_kin(i):
        o = objs[i]
        out_ = set()
        for (r, c) in o.cells:
            for (dr, dc) in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                j = cell_owner.get((r + dr, c + dc))
                if j is not None and j != i and objs[j].size <= 2 and objs[j].color == o.color:
                    out_.add(j)
        return out_
    # (a LINE: each piece meets at most two others corner to corner; a
    # checkered field, where each meets four, is a texture, not a line)
    for i, o in enumerate(objs):
        if o.size > 2 or o.color in ground or gsize[find(i)] > 2:
            continue
        kin_i = diag_kin(i)
        if len(kin_i) > 2:
            continue
        for j in kin_i:
            if gsize[find(j)] > 2 or len(diag_kin(j)) > 2:
                continue
            union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    H, W = logical.shape
    for g in groups.values():
        cells = {}
        for i in g:
            for p in objs[i].cells:
                cells[p] = int(objs[i].color)
        s = Sprite(sid=-1, cells=cells, parts=len(g))
        r0, c0, r1, c1 = s.bbox
        thin_wide = (s.dims[0] <= 3 and s.dims[1] >= 12) or (s.dims[1] <= 3 and s.dims[0] >= 12)
        at_edge = r0 <= 1 or r1 >= H - 2 or c0 <= 1 or c1 >= W - 2
        if thin_wide and at_edge and s.is_bar():
            s.tags.add("bar")
        if r0 in hud_rows or r1 in hud_rows:
            s.tags.add("hud")
        if s.size >= BIG:
            s.tags.add("terrain")
        out.append(s)
    return tag_textures(out, logical, bg)


def read_components(logical, bg):
    """Every same-colour component as its own sprite (parts=1): the unit
    at which effects are learned — a base stays, a bar grows, a tip
    rides, and a rigid sprite cannot say so."""
    objs = segment(logical, bg, mask=None)
    H, W = logical.shape
    out = []
    area = Counter()
    for o in objs:
        area[o.color] += o.size
    for o in objs:
        s = Sprite(sid=-1, cells={p: int(o.color) for p in o.cells}, parts=1)
        r0, c0, r1, c1 = s.bbox
        thin_wide = (s.dims[0] <= 3 and s.dims[1] >= 12) or (s.dims[1] <= 3 and s.dims[0] >= 12)
        at_edge = r0 <= 1 or r1 >= H - 2 or c0 <= 1 or c1 >= W - 2
        if thin_wide and at_edge and s.is_bar():
            s.tags.add("bar")
        # a thin thing lying on the frame's first or last row is HUD (a
        # step bar's spent and remaining halves, a life counter)
        if s.dims[0] <= 3 and (r0 == 0 or r1 == H - 1):
            s.tags.add("hud")
        if s.size >= BIG or area[o.color] >= 300:
            s.tags.add("terrain")
        out.append(s)
    # A STRIP: four or more small things of one shape in a line at an
    # even pitch along a frame edge (a row of dashes, a rank of hearts,
    # a legend) is a display — a budget, a count — not a thing in play,
    # whatever colour each dash has
    for strip in hud_strips(out, H, W):
        for s in strip:
            s.tags.add("hud")
    return tag_textures(out, logical, bg)


def hud_strips(sprites, H, W, margin=4):
    small = [s for s in sprites if s.size <= 12 and "terrain" not in s.tags]
    by_shape = defaultdict(list)
    for s in small:
        by_shape[s.shape].append(s)
    strips = []
    for shape, ss in by_shape.items():
        if len(ss) < 4:
            continue
        for axis in (0, 1):            # a row (same r0) or a column (same c0)
            lines = defaultdict(list)
            for s in ss:
                lines[s.bbox[axis]].append(s)
            for pos, line in lines.items():
                if len(line) < 4:
                    continue
                edge_near = pos <= margin or (pos + (line[0].dims[axis] - 1)) >= (H if axis == 0 else W) - 1 - margin
                if not edge_near:
                    continue
                line.sort(key=lambda s: s.bbox[1 - axis])
                gaps = [line[i + 1].bbox[1 - axis] - line[i].bbox[1 - axis] for i in range(len(line) - 1)]
                if gaps and max(gaps) - min(gaps) <= 1 and min(gaps) >= line[0].dims[1 - axis]:
                    strips.append(line)
    return strips


# ------------------------------------------------------------- tracking

@dataclass
class Event:
    kind: str            # moved rotated recoloured grew shrank appeared vanished merged split
    sid: int
    data: dict = field(default_factory=dict)

    def __repr__(self):
        return f"{self.kind}(S{self.sid} {self.data})"


def _emblems(sprites):
    """For every sprite, the colours of the tiny sprites (<= 3 cells)
    it encloses — a block's centre dot.  A distinguishing feature that
    travels with its wearer: identity prefers the same emblem."""
    tiny = [t for t in sprites if t.size <= 3]
    out = {id(s): set() for s in sprites}
    for t in tiny:
        own = set(t.cells)
        for h in sprites:
            if h is t or h.size <= 3:
                continue
            hb = h.bbox
            if not (hb[0] <= t.bbox[0] and hb[1] <= t.bbox[1] and hb[2] >= t.bbox[2] and hb[3] >= t.bbox[3]):
                continue
            if all(((r + dr, c + dc) in h.cells or (r + dr, c + dc) in own)
                   for r, c in own for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                out[id(h)].add(t.body)
    for s in sprites:
        s.emblem = frozenset(out[id(s)])
    return out


import os as _os
_ENV_NO_SWAP = bool(_os.environ.get("NO_SWAP_RULE"))


class EntityTracker:
    """Identity across frames and the event diff per transition."""

    def __init__(self):
        self.next_id = 1
        self.prev = []               # sprites of the previous frame (with ids)
        self.split_keys = set()      # (colour, colour) pairs proven not rigid
        self.history = {}            # sid -> list of (centre, size) — recent
        self.last_events = []
        self.covered = {}            # sid -> sprite last seen, now under a mover

    def _new(self, s):
        s.sid = self.next_id; self.next_id += 1
        return s

    def update(self, sprites):
        """Assign identities to the sprites of a new frame; return events."""
        prev = self.prev
        events = []
        if not prev:
            for s in sprites:
                self._new(s)
            self.prev = sprites
            self.last_events = events
            return events
        unmatched_prev = {p.sid: p for p in prev}
        cur_taken = set()
        _emblems(sprites)
        # UNCOVERED: a thing that vanished under a mover and now shows
        # again, the same cells, the same look, is that thing — it never
        # moved (a tile under a walker; a node under a token).  Settled
        # before any matching, or the grid seems to slide behind the mover
        for s in sprites:
            for csid, cp in list(self.covered.items()):
                if s.key == cp.key and set(s.cells) == set(cp.cells):
                    s.sid = csid; cur_taken.add(id(s))
                    del self.covered[csid]
                    events.append(Event("uncovered", csid, {}))
                    break
        for p in prev:
            if not hasattr(p, "emblem"):
                p.emblem = frozenset()

        def dist(p, s):
            d = abs(s.centre[0] - p.centre[0]) + abs(s.centre[1] - p.centre[1])
            return d
        # TWO THINGS THAT SWAPPED COLOURS STAYED WHERE THEY WERE: a previous
        # thing's exact cells now wear another previous thing's colours, and
        # that other thing's exact cells wear the first's — a selection
        # shown by the whole body's colour (sp80 L2: a click makes the
        # clicked bar wear the chosen colour and the chosen bar the plain
        # one; read as "the chosen bar jumped", every law after it was of a
        # bar that never moved).  Nothing passed through anything: both
        # recoloured in place.
        if not _ENV_NO_SWAP:
            cur_by_cells = {}
            for s in sprites:
                if id(s) not in cur_taken and s.size >= 4:
                    cur_by_cells[frozenset(s.cells.keys())] = s
            prev_by_cells = {}
            for p in prev:
                if p.sid in unmatched_prev and p.size >= 4 and "terrain" not in p.tags and "hud" not in p.tags:
                    prev_by_cells[frozenset(p.cells.keys())] = p
            for X, p in list(prev_by_cells.items()):
                c = cur_by_cells.get(X)
                if _os.environ.get("SWAP_DEBUG") and c is not None and c.key != p.key:
                    print(f"   [swap] S{p.sid} {p.body}/{p.size} cells now wear {c.body}: taken {id(c) in cur_taken} unmatched {p.sid in unmatched_prev} shape {c.shape == p.shape}")
                if c is None or id(c) in cur_taken or p.sid not in unmatched_prev or c.key == p.key or c.shape != p.shape:
                    continue
                # the partner: a previous thing wearing c's colours whose cells
                # now wear p's
                q = next((q_ for Y, q_ in prev_by_cells.items() if q_.sid != p.sid and q_.sid in unmatched_prev and q_.key == c.key
                          and cur_by_cells.get(Y) is not None and id(cur_by_cells[Y]) not in cur_taken and cur_by_cells[Y].key == p.key and cur_by_cells[Y].shape == q_.shape), None)
                if q is None:
                    continue
                d = cur_by_cells[frozenset(q.cells.keys())]
                for pp, cc in ((p, c), (q, d)):
                    cc.sid = pp.sid; cur_taken.add(id(cc))
                    del unmatched_prev[pp.sid]
                    events.append(Event("recoloured", pp.sid, {"n": cc.size, "livery": sorted(cc.colors), "swap": True}))
        # pass 1: same class, nearest — exact identity (static or moved)
        by_key = defaultdict(list)
        for s in sprites:
            if id(s) not in cur_taken:
                by_key[s.key].append(s)
        # NOTHING PASSES THROUGH ANOTHER, AND A DOT'S COLOUR IS CHEAP: among
        # the assignments of a small class of same-body things (twins that
        # may differ by their emblem or by a transition frame's shape),
        # the one whose largest displacement is smallest; then the least
        # total; then the fewest emblem changes.  Two touching twins pushed
        # by 3 each, not one still and one jumping 6; two twins whose dots
        # swap colour under a click stay where they are.
        import itertools
        prev_by_body = defaultdict(list)
        for p in prev:
            if p.size >= 4 and "terrain" not in p.tags and "hud" not in p.tags:
                prev_by_body[p.body].append(p)
        cur_by_body = defaultdict(list)
        for s in sprites:
            if s.size >= 4 and "terrain" not in s.tags and "hud" not in s.tags and id(s) not in cur_taken:
                cur_by_body[s.body].append(s)
        # UNDER SOMETHING NOW: a previous thing whose cells are all held by
        # a thing of another class in this frame is covered, not moved —
        # leaving it unmatched costs nothing
        cell_owner = {}
        for s in sprites:
            for q in s.cells:
                cell_owner[q] = s
        under = set()
        import os
        class_n = Counter(p.key for p in prev) if not os.environ.get("NOUNDER") else Counter()
        prev_by_key_all = defaultdict(list)
        for p in prev:
            prev_by_key_all[p.key].append(p)
        prev_owned = set()
        for p in prev:
            prev_owned |= set(p.cells)
        for p in prev:
            # a unique thing that shows again nearby has moved; only a
            # thing with twins (a grid's tiles) can be the one that stayed
            if "terrain" in p.tags or "hud" in p.tags or class_n[p.key] < 3:
                continue
            owners = {id(cell_owner[q]) for q in p.cells if q in cell_owner}
            # the coverer must be a distinguished thing (a walker, an
            # arm) — a twin among twins (a tile shifted along a full
            # track) covers nothing, it took the slot
            cur_n = Counter(s.key for s in sprites)

            def moved_in(s):
                # the coverer came from elsewhere: a thing of its exact
                # class stood in the last frame clear of p (a walker);
                # a thing that merely grew into p's cells (a track
                # showing through a vacated slot) covers nothing
                return any(q.key == s.key and not (set(q.cells) & set(p.cells)) for q in prev_by_key_all.get(s.key, []))
            if owners and all(q in cell_owner for q in p.cells) and all(cell_owner[q].key != p.key for q in p.cells) \
                    and all(cur_n[cell_owner[q].key] < 3 and moved_in(cell_owner[q]) for q in p.cells):
                # ...unless a twin now stands nearby on ground that was
                # EMPTY before: then this thing moved there (a tip that a
                # bar grew over sits at the bar's new end); a twin on
                # ground a mover just left is an uncovered tile
                moved_there = False
                for s in by_key.get(p.key, []):
                    d = abs(s.centre[0] - p.centre[0]) + abs(s.centre[1] - p.centre[1])
                    if 0 < d <= 24 and not any(q in prev_owned for q in s.cells):
                        moved_there = True; break
                if not moved_there:
                    under.add(p.sid)
        # ...and unless it rode a shift that others rode too: when two or
        # more other things moved by exactly the vector that would carry
        # this thing onto its twin, it moved with them (a full track
        # turning one slot: every piece's slot is taken by the next)
        if under:
            votes = Counter()
            for p in prev:
                if p.sid in under or "terrain" in p.tags or "hud" in p.tags:
                    continue
                best_s, best_d = None, 13
                for s in by_key.get(p.key, []):
                    d = abs(s.centre[0] - p.centre[0]) + abs(s.centre[1] - p.centre[1])
                    if 0 < d < best_d:
                        best_s, best_d = s, d
                if best_s is not None:
                    votes[(best_s.centre[0] - p.centre[0], best_s.centre[1] - p.centre[1])] += 1
            for p in prev:
                if p.sid not in under:
                    continue
                for s in by_key.get(p.key, []):
                    v = (s.centre[0] - p.centre[0], s.centre[1] - p.centre[1])
                    if votes.get(v, 0) >= 2:
                        under.discard(p.sid); break
        forced = {}
        for body, ps in prev_by_body.items():
            ss = cur_by_body.get(body, [])
            if not (2 <= len(ps) <= 5 and 2 <= len(ss) <= 5):
                continue
            import math
            best = None
            # each previous thing takes a current one or none (None: it
            # vanished or turned into something else — as costly as a
            # jump of 20); a size change costs like a displacement
            slots = list(range(len(ss))) + [None] * len(ps)
            for sub in itertools.permutations(slots, len(ps)):
                ds = []; emb = 0
                for i, j in enumerate(sub):
                    p = ps[i]
                    if j is None:
                        ds.append(0.0 if p.sid in under else 20.0); continue
                    c = ss[j]
                    d = abs(c.centre[0] - p.centre[0]) + abs(c.centre[1] - p.centre[1])
                    d += 8.0 * abs(math.log2(c.size / p.size))
                    ds.append(d)
                    emb += p.emblem != c.emblem
                cost = (max(ds), sum(ds), emb)
                if best is None or cost < best[0]:
                    best = (cost, sub)
            if best is not None:
                for i, j in enumerate(best[1]):
                    if j is not None and 0.5 * ps[i].size <= ss[j].size <= 2.0 * ps[i].size:
                        forced[ps[i].sid] = id(ss[j])
        # a forced pair of another shape (a transition frame) is settled
        # here; same-shape pairs go through the nearest-first pass below
        for p in prev:
            if p.sid in forced:
                s = next(x for x in sprites if id(x) == forced[p.sid])
                if s.key == p.key:
                    continue
                s.sid = p.sid; cur_taken.add(id(s)); del unmatched_prev[p.sid]
                vec = (s.centre[0] - p.centre[0], s.centre[1] - p.centre[1])
                events.append(Event("grew" if s.size > p.size else "shrank", p.sid, {"delta": s.size - p.size, "vec": vec}))
                if abs(vec[0]) + abs(vec[1]) > 0.5:
                    events.append(Event("moved", p.sid, {"vec": vec}))
        pairs = []
        for p in prev:
            if p.sid in under:
                continue            # covered where it stood: it did not move
            for s in by_key.get(p.key, []):
                d = dist(p, s)
                if p.sid in forced and forced[p.sid] != id(s):
                    continue
                pairs.append((d, p.sid, id(s), p, s))
        pairs.sort(key=lambda t: t[0])
        for d, psid, sid_, p, s in pairs:
            if psid not in unmatched_prev or id(s) in cur_taken:
                continue
            if d > 40 and len(by_key[p.key]) > 1:
                continue    # far and ambiguous
            s.sid = psid; cur_taken.add(id(s)); del unmatched_prev[psid]
            vec = (s.centre[0] - p.centre[0], s.centre[1] - p.centre[1])
            if abs(vec[0]) + abs(vec[1]) > 0:
                events.append(Event("moved", psid, {"vec": vec}))
        # pass 2: same body class (livery changed) or rotated / grown /
        # recoloured versions of an unmatched previous sprite, nearest
        rest = [s for s in sprites if id(s) not in cur_taken]
        for psid, p in list(unmatched_prev.items()):
            if psid in under:
                continue
            best, bd, why = None, 1e9, None
            pshape = p.shape
            dih = None
            for s in rest:
                if id(s) in cur_taken:
                    continue
                d = dist(p, s)
                if s.shape == pshape and s.body == p.body:
                    w = ("recoloured", d)
                elif s.body == p.body and d <= 24 and (
                        0.5 * p.size <= s.size <= 2.0 * p.size
                        or set(p.cells) <= set(s.cells) or set(s.cells) <= set(p.cells)):
                    # a bar that grew from two cells to five is the same
                    # bar: its old cells are still there
                    if dih is None:
                        dih = dihedral(pshape)
                    if s.shape in dih.values():
                        w = ("rotated", d + 2)
                    else:
                        w = ("grew" if s.size > p.size else "shrank", d + 4)
                elif s.shape == pshape and d <= 6:
                    w = ("recoloured", d + 6)
                else:
                    continue
                if w[1] < bd:
                    best, bd, why = s, w[1], w[0]
            if best is not None and bd <= 20:
                best.sid = psid; cur_taken.add(id(best)); del unmatched_prev[psid]
                vec = (best.centre[0] - p.centre[0], best.centre[1] - p.centre[1])
                if why == "rotated":
                    events.append(Event("rotated", psid, {"vec": vec}))
                elif why == "recoloured":
                    changed = [q for q in best.cells if p.cells.get((q[0] - int(round(vec[0])), q[1] - int(round(vec[1])))) != best.cells[q]]
                    events.append(Event("recoloured", psid, {"n": len(changed), "livery": sorted(best.colors)}))
                else:
                    events.append(Event(why, psid, {"delta": best.size - p.size, "vec": vec}))
                if abs(vec[0]) + abs(vec[1]) > 0.5 and why != "recoloured":
                    events.append(Event("moved", psid, {"vec": vec}))
        # pass 3: merges — an unmatched previous sprite whose cells now lie
        # inside a matched or new sprite
        for psid, p in list(unmatched_prev.items()):
            host = None
            for s in sprites:
                if "terrain" in s.tags:
                    continue
                if sum(1 for q in p.cells if q in s.cells) * 2 >= p.size and s.size > p.size:
                    host = s; break
            if host is not None:
                if host.sid < 0:
                    self._new(host)
                events.append(Event("merged", psid, {"into": host.sid}))
                del unmatched_prev[psid]
        # pass 4: splits — a new sprite whose cells were part of one
        # previous sprite that is still present (smaller)
        present = {p.sid for p in prev if p.sid not in unmatched_prev}
        for s in sprites:
            if s.sid >= 0:
                continue
            # a split needs a donor that is STILL THERE (smaller); a new
            # thing over a vanished one is a change of form, handled below
            donors = {p.sid for p in prev if p.sid in present and any(q in p.cells for q in s.cells)}
            self._new(s)
            if len(donors) == 1:
                d = donors.pop()
                events.append(Event("split", d, {"child": s.sid}))
                # the parts never were rigid: never glue these colours again
                donor = next(p for p in prev if p.sid == d)
                for a in set(s.cells.values()):
                    for b in set(donor.cells.values()):
                        if a != b:
                            self.split_keys.add((a, b)); self.split_keys.add((b, a))
            else:
                events.append(Event("appeared", s.sid, {"size": s.size, "body": s.body}))
        # ONE THING, CHANGED IN FORM: a thing that vanished and a thing that
        # appeared in the same transition, near each other, are one thing
        # that jumped and changed (a dot that became a block at the click);
        # nearest pairs first, within 24 cells, never across terrain/HUD
        appeared = [e for e in events if e.kind == "appeared"]
        if appeared and unmatched_prev:
            new_by_sid = {s.sid: s for s in sprites}
            cands = []
            for psid, p in unmatched_prev.items():
                if "terrain" in p.tags or "hud" in p.tags or psid in under:
                    continue
                for e in appeared:
                    s = new_by_sid[e.sid]
                    if "terrain" in s.tags or "hud" in s.tags:
                        continue
                    d = abs(s.centre[0] - p.centre[0]) + abs(s.centre[1] - p.centre[1])
                    if d <= 24:
                        cands.append((d, psid, e.sid))
            cands.sort()
            used_p, used_s = set(), set()
            for d, psid, nsid in cands:
                if psid in used_p or nsid in used_s:
                    continue
                used_p.add(psid); used_s.add(nsid)
                s = new_by_sid[nsid]; p = unmatched_prev[psid]
                s.sid = psid
                events = [e for e in events if not (e.kind == "appeared" and e.sid == nsid)]
                vec = (s.centre[0] - p.centre[0], s.centre[1] - p.centre[1])
                events.append(Event("morphed", psid, {"vec": vec, "delta": s.size - p.size, "body": s.body, "was": p.body}))
                if abs(vec[0]) + abs(vec[1]) > 0.5:
                    events.append(Event("moved", psid, {"vec": vec}))
            for psid in used_p:
                del unmatched_prev[psid]
        # COVERED: a thing that vanished where a moved thing now lies is
        # under it — remembered until it shows again
        moved_now = {e.sid for e in events if e.kind == "moved"}
        over = [s for s in sprites if s.sid in moved_now and "terrain" not in s.tags]
        for psid, p in unmatched_prev.items():
            pc = set(p.cells)
            if psid in under or any(pc <= set(s.cells) for s in over):
                self.covered[psid] = p
                events.append(Event("covered", psid, {"size": p.size, "body": p.body}))
            else:
                events.append(Event("vanished", psid, {"size": p.size, "body": p.body}))
        self.prev = sprites
        self.last_events = events
        return events

    def get(self, sid):
        for s in self.prev:
            if s.sid == sid:
                return s
        return None


# ------------------------------------------------------------- classes

def classes(sprites):
    """Class key -> members (same shape, same colours)."""
    out = defaultdict(list)
    for s in sprites:
        out[s.key].append(s)
    return out


def describe(sprites, limit=40):
    rows = []
    for s in sorted(sprites, key=lambda s: (-s.size))[:limit]:
        rows.append(f"  {s!r} parts={s.parts} livery={len(s.livery)} cavity={len(s.cavity())} tags={sorted(s.tags)}")
    return "\n".join(rows)
