"""THE LEGEND WITHIN: a small patterned thing at the centre of a lattice of
like tiles is a miniature of its surround — each of its cells stands for
the tile in that direction, and its colours name the tiles' colours.

Read at two scales (Tushar, ft09): the panel as one thing — a glyph and
the ring of tiles around it — and the relation between the glyph's cells
and the ring.  Panels that agree under one mapping (glyph colour -> tile
colour) are the examples; the panel that breaks it is the puzzle, and the
mapping says what each of its tiles must become.  The mapping is carried
across levels as the working hypothesis: on a later level there may be no
example, only cards, and a tile between two cards must satisfy both.

Nothing here names a colour, a size or a place: the tiles are whatever
single-colour things of one size lie on a lattice; the glyphs are whatever
multi-colour things of that size sit at lattice points among them; the
pitch is what the lattice shows.
"""
from collections import Counter, defaultdict
import numpy as np


def _components(frame, bg):
    """4-connected regions of ONE colour (not the background): a tile is a
    patch of its colour whatever surrounds it — a panel's box, the ground."""
    H, W = frame.shape
    seen = np.zeros((H, W), dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(frame[r, c]) == bg:
                continue
            col = int(frame[r, c])
            cells = []; fr = [(r, c)]; seen[r, c] = True
            while fr:
                y, x = fr.pop(); cells.append((y, x))
                for y2, x2 in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= y2 < H and 0 <= x2 < W and not seen[y2, x2] and int(frame[y2, x2]) == col:
                        seen[y2, x2] = True; fr.append((y2, x2))
            out.append(cells)
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def read_panels(frame, bg, hud_rows=()):
    """Find the lattice of tiles and the glyphs among them.  Returns
    (panels, pitch, tile_dims): each panel = {"centre": (r, c), "glyph":
    {(dy, dx): colour}, "ring": {(dy, dx): (colour, (r, c))}} for the 8
    directions dy, dx in -1..1, ring entries only where a tile lies."""
    H, W = frame.shape
    comps = _components(frame, bg)
    boxes = []
    for cells in comps:
        b = _bbox(cells)
        if b[0] in hud_rows or b[2] in hud_rows:
            continue
        h, w = b[2] - b[0] + 1, b[3] - b[1] + 1
        cols = Counter(int(frame[r, c]) for r, c in cells)
        boxes.append((b, h, w, cols, cells))
    # TILES: full rectangles of one colour, all of one size, many of them
    # (a tile: a patch of one colour filling most of its box — a notch or a
    # mark drawn on it in another colour does not make it less a tile)
    solid = [(b, h, w, cols) for (b, h, w, cols, cells) in boxes
             if len(cols) == 1 and len(cells) >= 0.7 * h * w and 2 <= h <= 16 and 2 <= w <= 16]
    if len(solid) < 6:
        return [], None, None
    # (the tiles' size: the one that covers most of the board — a legend's
    # own cells are small solid blocks too, but few and small)
    area = Counter()
    for _, h, w, _ in solid:
        area[(h, w)] += h * w
    dims = max(area, key=area.get)
    tiles = [(b, next(iter(cols))) for (b, h, w, cols) in solid if (h, w) == dims]
    if len(tiles) < 6:
        return [], None, None
    centres = {((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0): col for b, col in tiles}
    # THE PITCH: the commonest axis-aligned distance between tile centres
    # (each tile's NEAREST tile along a row or a column: a lattice with
    # gaps still shows its pitch in the nearest pairs; all pairs would
    # count the doubled distances across the gaps as often)
    ds = Counter()
    pts = list(centres)
    for (r1, c1) in pts:
        for axis in (0, 1):
            near = [abs((r2, c2)[axis] - (r1, c1)[axis]) for (r2, c2) in pts
                    if abs((r2, c2)[1 - axis] - (r1, c1)[1 - axis]) < 0.5 and abs((r2, c2)[axis] - (r1, c1)[axis]) > 0.5]
            if near:
                ds[min(near)] += 1
    if not ds:
        return [], None, None
    pitch = min(ds, key=lambda d: (-ds[d], d))
    # GLYPHS: at the lattice points among the tiles, a box of the tiles'
    # size showing several colours (a miniature: its cells in a k x k
    # grid, k = 3 for the eight neighbours)
    panels = []
    h, w = dims
    k = 3
    if h % k or w % k:
        return [], pitch, dims
    ch, cw = h // k, w // k
    cand = set()
    for (r0, c0) in pts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (r0 + dy * pitch, c0 + dx * pitch)
                if not any(abs(p[0] - q[0]) < 0.6 and abs(p[1] - q[1]) < 0.6 for p in pts):
                    cand.add((round(q[0] * 2) / 2.0, round(q[1] * 2) / 2.0))
    seen_b = set()
    for (qr, qc) in cand:
        b = (int(round(qr - (h - 1) / 2.0)), int(round(qc - (w - 1) / 2.0)), 0, 0)
        b = (b[0], b[1], b[0] + h - 1, b[1] + w - 1)
        if b[0] < 0 or b[1] < 0 or b[2] >= H or b[3] >= W or b in seen_b:
            continue
        seen_b.add(b)
        cols = {int(frame[r, c]) for r in range(b[0], b[2] + 1) for c in range(b[1], b[3] + 1)}
        if len(cols) < 2 or bg in cols:
            continue                    # (a miniature is drawn whole, in its own colours)
        glyph = {}
        ok = True
        for dy in range(k):
            for dx in range(k):
                blk = {int(frame[r, c]) for r in range(b[0] + dy * ch, b[0] + (dy + 1) * ch)
                       for c in range(b[1] + dx * cw, b[1] + (dx + 1) * cw)}
                if len(blk) != 1:
                    ok = False; break
                glyph[(dy - 1, dx - 1)] = next(iter(blk))
            if not ok:
                break
        if not ok:
            continue
        # (a tile drawn with a pattern in its own colour — a cross that
        # toggles its neighbours — is a tile, not a legend: a legend's
        # cells wear colours no tile wears, its centre apart)
        tile_cols = {col for _, col in tiles}
        if any(col in tile_cols for d_, col in glyph.items() if d_ != (0, 0)):
            continue
        cr, cc = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        ring = {}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (cr + dy * pitch, cc + dx * pitch)
                hit = next(((p, col) for p, col in centres.items() if abs(p[0] - q[0]) < 0.6 and abs(p[1] - q[1]) < 0.6), None)
                if hit is not None:
                    ring[(dy, dx)] = (hit[1], (int(round(hit[0][0])), int(round(hit[0][1]))))
                    continue
                # (a tile drawn with a pattern — most of its box in one tile
                # colour — is a tile of that colour)
                qb = (int(round(q[0] - (h - 1) / 2.0)), int(round(q[1] - (w - 1) / 2.0)))
                if 0 <= qb[0] and qb[0] + h <= H and 0 <= qb[1] and qb[1] + w <= W:
                    blk = Counter(int(frame[r, c]) for r in range(qb[0], qb[0] + h) for c in range(qb[1], qb[1] + w))
                    top_col, top_n = blk.most_common(1)[0]
                    if top_col in tile_cols and top_n >= 0.5 * h * w:
                        ring[(dy, dx)] = (top_col, (int(round(q[0])), int(round(q[1]))))
        if len(ring) >= 3:
            panels.append({"centre": (cr, cc), "glyph": glyph, "ring": ring, "box": b})
    return panels, pitch, dims


def learn_map(panels):
    """The mapping glyph colour -> tile colours, from the panels: each
    panel's ring, read against its glyph.  A colour of the glyph that
    stands for the glyph's OWN centre colour is named as such (the centre
    is the card's colour; "same as the centre" carries to a card of another
    colour), the others as "another colour than the centre"."""
    votes = defaultdict(Counter)
    for p in panels:
        centre_col = p["glyph"].get((0, 0))
        for d, (tcol, _) in p["ring"].items():
            g = p["glyph"].get(d)
            if g is None:
                continue
            rel = "same" if tcol == centre_col else "other"
            votes[g][rel] += 1
    return {g: cnt.most_common(1)[0][0] for g, cnt in votes.items()}, votes


def consistent(panel, mapping):
    """Whether every ring tile of the panel obeys the mapping."""
    centre_col = panel["glyph"].get((0, 0))
    for d, (tcol, _) in panel["ring"].items():
        g = panel["glyph"].get(d)
        rel = mapping.get(g)
        if rel is None:
            continue
        if (rel == "same") != (tcol == centre_col):
            return False
    return True


def violations(panels, mapping):
    """Tiles that break the mapping, with what they must become: a list of
    (tile centre, current colour, wanted: ("same", centre colour) or
    ("other", centre colour)).  A tile between two cards must satisfy
    each: the demands are gathered per tile."""
    demands = defaultdict(list)
    for p in panels:
        centre_col = p["glyph"].get((0, 0))
        for d, (tcol, pos) in p["ring"].items():
            rel = mapping.get(p["glyph"].get(d))
            if rel is None:
                continue
            demands[pos].append((rel, centre_col, tcol))
    out = []
    for pos, ds in demands.items():
        tcol = ds[0][2]
        bad = [(rel, cc) for rel, cc, _ in ds if (rel == "same") != (tcol == cc)]
        if bad:
            out.append((pos, tcol, [(rel, cc) for rel, cc, _ in ds]))
    return out


def satisfies(colour, demands):
    return all((rel == "same") == (colour == cc) for rel, cc in demands)


def tile_look(frame, pos, dims):
    """How a tile is drawn: the offsets within its box that are not its
    main colour (a plain tile: none; a cross: its arms).  Tiles that look
    alike act alike — one click on one of them teaches all."""
    h, w = dims
    r0, c0 = pos[0] - (h - 1) // 2, pos[1] - (w - 1) // 2
    blk = Counter(int(frame[r, c]) for r in range(r0, r0 + h) for c in range(c0, c0 + w))
    main = blk.most_common(1)[0][0]
    return main, frozenset((r - r0, c - c0) for r in range(r0, r0 + h) for c in range(c0, c0 + w) if int(frame[r, c]) != main)


def solve_flips(need, effects, order):
    """Which tiles to click so that exactly the tiles in `need` change
    (two colours: a click toggles; two clicks undo).  `effects[pos]` = the
    set of tile positions a click on pos toggles.  Gaussian elimination
    over GF(2); returns a list of positions, or None when no set of clicks
    does it."""
    idx = {p: i for i, p in enumerate(order)}
    n = len(order)
    rows = []                       # one equation per tile: sum of clicks toggling it = need
    for t in order:
        mask = 0
        for p in order:
            if t in effects.get(p, ()):
                mask |= 1 << idx[p]
        rows.append([mask, 1 if t in need else 0])
    # eliminate
    pivots = []
    r = 0
    for col in range(n):
        piv = next((i for i in range(r, len(rows)) if rows[i][0] >> col & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][0] >> col & 1:
                rows[i][0] ^= rows[r][0]; rows[i][1] ^= rows[r][1]
        pivots.append((r, col)); r += 1
    for i in range(r, len(rows)):
        if rows[i][0] == 0 and rows[i][1]:
            return None             # inconsistent
    x = 0
    for (ri, col) in pivots:
        if rows[ri][1]:
            x |= 1 << col
    return [order[i] for i in range(n) if x >> i & 1]
