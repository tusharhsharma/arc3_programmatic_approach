"""A TABLE OF PAIRS: two things joined by a connector are a pair, and a
column of such pairs is a table that maps forms to forms.  Below it, a
row of forms is a question and the row under it the answer, position by
position; a cursor marks the position being answered.

The principles are the core's own, not this game's: a form is the same
form turned or mirrored (Tushar: "understand object rotate, mirror and
see what matches"); what a connector joins belongs together; like
things in a row are read as a sequence.  Nothing here names a colour, a
size or a place.
"""
from collections import Counter, defaultdict
import numpy as np


def _components(frame, colour):
    """8-connected components of one colour: lists of cells."""
    H, W = frame.shape
    mask = frame == colour
    seen = np.zeros_like(mask)
    out = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            cells = []; fr = [(r, c)]; seen[r, c] = True
            while fr:
                y, x = fr.pop(); cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        y2, x2 = y + dy, x + dx
                        if 0 <= y2 < H and 0 <= x2 < W and mask[y2, x2] and not seen[y2, x2]:
                            seen[y2, x2] = True; fr.append((y2, x2))
            out.append(cells)
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def canon(cells):
    """The form of a set of cells, the same under the eight turns and
    mirrors: the least of its eight drawings, each set at its own corner."""
    pts = [(r, c) for r, c in cells]
    best = None
    for k in range(8):
        q = pts
        if k >= 4:
            q = [(r, -c) for r, c in q]
        for _ in range(k % 4):
            q = [(c, -r) for r, c in q]
        r0 = min(r for r, _ in q); c0 = min(c for _, c in q)
        key = tuple(sorted((r - r0, c - c0) for r, c in q))
        if best is None or key < best:
            best = key
    return best


def _glyph_clusters(cells):
    """Glyphs within a strip of tiles: the glyph colour's components,
    merged when their boxes lie within a cell of each other (a glyph may
    be drawn in pieces; two glyphs stand two cells apart or more)."""
    comps = []
    H = max(r for r, _ in cells) + 2; W = max(c for _, c in cells) + 2
    m = np.zeros((H, W), dtype=np.int32)
    for r, c in cells:
        m[r, c] = 1
    for comp in _components(m, 1):
        comps.append(comp)
    merged = True
    while merged:
        merged = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = _bbox(comps[i]), _bbox(comps[j])
                gap = max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])
                if gap <= 1:
                    comps[i] = comps[i] + comps[j]; del comps[j]; merged = True; break
            if merged:
                break
    return comps


def read_table(frame, bg, hud_rows=()):
    """Tiles: regions of one colour whose boxes hold one other colour's
    marks (the glyphs).  Returns a dict with the glyphs (each: colour of
    tile, cells, box, form), the pairs (glyph index, glyph index) joined
    by a connector, the rows (aligned glyphs of one tile colour not in a
    pair) and the cursor's position among the answer row, or None."""
    H, W = frame.shape
    cols = Counter(int(x) for x in frame.ravel())
    glyphs = []                                   # {"tile": colour, "cells", "box", "form"}
    tile_cols = set()
    for tcol in cols:
        if tcol == bg:
            continue
        for comp in _components(frame, tcol):
            b = _bbox(comp)
            if b[0] in hud_rows or b[2] in hud_rows:
                continue
            h, w = b[2] - b[0] + 1, b[3] - b[1] + 1
            if min(h, w) < 5 or len(comp) < 0.4 * h * w:
                continue
            # (a tile FRAMES its glyph: the box's rim is the tile's colour throughout)
            rim = [(b[0], c) for c in range(b[1], b[3] + 1)] + [(b[2], c) for c in range(b[1], b[3] + 1)] + \
                  [(r, b[1]) for r in range(b[0], b[2] + 1)] + [(r, b[3]) for r in range(b[0], b[2] + 1)]
            if any(int(frame[r, c]) != tcol for r, c in rim):
                continue
            inside = Counter(int(frame[r, c]) for r in range(b[0], b[2] + 1) for c in range(b[1], b[3] + 1))
            others = {c_: n_ for c_, n_ in inside.items() if c_ != tcol}
            if len(others) != 1:
                continue
            gcol, n_g = next(iter(others.items()))
            if n_g < 3 or n_g > 0.6 * h * w:
                continue
            gcells = [(r, c) for r in range(b[0], b[2] + 1) for c in range(b[1], b[3] + 1) if int(frame[r, c]) == gcol]
            for cl in _glyph_clusters(gcells):
                if len(cl) < 3:
                    continue
                glyphs.append({"tile": tcol, "glyph_colour": gcol, "cells": cl, "box": _bbox(cl), "form": canon(cl), "tile_box": b,
                               "centre": (sum(r for r, _ in cl) / len(cl), sum(c for _, c in cl) / len(cl))})
            tile_cols.add(tcol)
    if len(glyphs) < 4:
        return None
    # (a glyph drawn with a hole reads as a "tile" of the glyph's colour
    # framing a scrap of the tile's: of two readings that frame each other,
    # the one seen more often is the tile)
    n_type = Counter((g["tile"], g["glyph_colour"]) for g in glyphs)
    glyphs = [g for g in glyphs if n_type[(g["glyph_colour"], g["tile"])] <= n_type[(g["tile"], g["glyph_colour"])]]
    tile_cols = {g["tile"] for g in glyphs}
    glyph_cols = {g["glyph_colour"] for g in glyphs}
    # CONNECTORS: thin short things of another colour whose two ends each
    # touch a tile's box — what they join is a pair
    pairs = []
    for ccol in cols:
        if ccol == bg or ccol in tile_cols or ccol in glyph_cols:
            continue
        for comp in _components(frame, ccol):
            if len(comp) > 24:
                continue
            b = _bbox(comp)
            h, w = b[2] - b[0] + 1, b[3] - b[1] + 1
            if min(h, w) > 2 or max(h, w) < 2:
                continue
            horiz = w >= h
            ends = [(b[0], b[1]), (b[2], b[3])] if not horiz else [(b[0], b[1]), (b[2], b[3])]
            e1 = (b[0] + (0 if horiz else 0), b[1]) ; e2 = (b[2], b[3])
            def nearest(pt, side):
                best = None
                for i, g in enumerate(glyphs):
                    gb = g["box"]
                    # the tile round the glyph is a cell or two wider than the glyph
                    if horiz:
                        ok = (gb[3] < pt[1] if side < 0 else gb[1] > pt[1]) and gb[0] - 2 <= pt[0] <= gb[2] + 2
                        d = (pt[1] - gb[3]) if side < 0 else (gb[1] - pt[1])
                    else:
                        ok = (gb[2] < pt[0] if side < 0 else gb[0] > pt[0]) and gb[1] - 2 <= pt[1] <= gb[3] + 2
                        d = (pt[0] - gb[2]) if side < 0 else (gb[0] - pt[0])
                    if ok and 0 < d <= 4 and (best is None or d < best[0]):
                        best = (d, i)
                return best[1] if best else None
            a = nearest(e1, -1); c = nearest(e2, +1)
            if a is not None and c is not None and a != c and glyphs[a]["tile"] != glyphs[c]["tile"]:
                pairs.append((a, c))
    paired = {i for p in pairs for i in p}
    # ROWS: aligned glyphs of one tile colour, not in a pair, three or more
    rows = []
    by = defaultdict(list)
    for i, g in enumerate(glyphs):
        if i in paired:
            continue
        by[(g["tile"], g["box"][0])].append(i)
    for (tcol, top), idxs in by.items():
        if len(idxs) >= 2:
            rows.append({"tile": tcol, "top": top, "glyphs": sorted(idxs, key=lambda i: glyphs[i]["box"][1])})
    rows.sort(key=lambda r: r["top"])
    # THE CURSOR: a small thing of a colour that is none of the above,
    # beside one glyph of a row
    cursor = None
    for ccol in cols:
        if ccol == bg or ccol in tile_cols or ccol in glyph_cols:
            continue
        for comp in _components(frame, ccol):
            if len(comp) > 30:
                continue
            b = _bbox(comp)
            for ri, row in enumerate(rows):
                for k, gi in enumerate(row["glyphs"]):
                    gb = glyphs[gi]["box"]; tb = glyphs[gi]["tile_box"]
                    # (beside the glyph's columns, just above or below its tile)
                    if b[1] <= gb[3] + 1 and b[3] >= gb[1] - 1 and (0 < tb[0] - b[2] <= 3 or 0 < b[0] - tb[2] <= 3):
                        cursor = (ri, k)
    return {"glyphs": glyphs, "pairs": pairs, "rows": rows, "cursor": cursor}


def mapping(table, from_col, to_col):
    """form -> form from the pairs whose tiles are of the two colours."""
    g = table["glyphs"]
    m = {}
    for a, b in table["pairs"]:
        if g[a]["tile"] == from_col and g[b]["tile"] == to_col:
            m[g[a]["form"]] = g[b]["form"]
        elif g[b]["tile"] == from_col and g[a]["tile"] == to_col:
            m[g[b]["form"]] = g[a]["form"]
    return m


def question_and_answer(table):
    """The two rows aligned one above the other — the question (a row of
    the table's left colour) and the answer (of its right colour) — and
    the mapping between them; None when the table shows no such rows."""
    rows = table["rows"]
    if len(rows) < 2:
        return None
    g = table["glyphs"]
    pair_cols = Counter((g[a]["tile"], g[b]["tile"]) for a, b in table["pairs"])
    if not pair_cols:
        return None
    (from_col, to_col), _ = pair_cols.most_common(1)[0]
    q = next((r for r in rows if r["tile"] == from_col), None)
    a = next((r for r in rows if r["tile"] == to_col), None)
    if q is None or a is None:
        return None
    cur = table.get("cursor")
    cursor = cur[1] if cur is not None and rows[cur[0]] is a else None
    return {"question": q, "answer": a, "map": mapping(table, from_col, to_col), "from": from_col, "to": to_col, "cursor": cursor}


def answer_status(table, qa):
    """For each answer position: (question form, wanted form, present form)."""
    g = table["glyphs"]
    out = []
    for k, ai in enumerate(qa["answer"]["glyphs"]):
        qi = qa["question"]["glyphs"][k] if k < len(qa["question"]["glyphs"]) else None
        want = qa["map"].get(g[qi]["form"]) if qi is not None else None
        out.append((g[qi]["form"] if qi is not None else None, want, g[ai]["form"]))
    return out
