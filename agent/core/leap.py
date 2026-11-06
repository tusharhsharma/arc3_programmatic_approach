"""A LEAP: select, then point — and what is leapt over is gone.

Tushar (lf52): "I saw that a new update came up in the board; clicking
that new position made one of the greens go away.  Hypothesis: eliminate
all the balls by jumping over them."

The principles composed here are the core's own: a click on a thing and
a click elsewhere is one experiment (select, then point — r11l); the
things of a kind on a lattice are read by their places; a thing that
vanished under a leap is removed by it; "remove all of a kind" is a goal
read from the first removal.  Nothing here names a colour, a size or a
place; identity across the leap is read from the places, not the ids
(the tracker may swap two like things that pass each other).
"""
import sys


def _centre(s):
    return s.centre


def _near(a, b, tol=1.5):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def read_leap(before, after, click_prev, click_now):
    """Two clicks, and the things of one kind before and after the second:
    the first click was on a thing A of the kind; after the second, A's
    place and a neighbour B's are empty and a thing of the kind stands
    at click_now, twice B's offset from A — a leap over B, B gone.
    Returns {"kind", "pitch", "from", "over", "to"} or None."""
    if click_now is None:
        return None
    under = [s for s in before if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags
             and click_prev is not None
             and s.bbox[0] <= click_prev[0] <= s.bbox[2] and s.bbox[1] <= click_prev[1] <= s.bbox[3]]
    if under:
        kinds = [(min(under, key=lambda s: s.size).body, min(under, key=lambda s: s.size).size)]
    else:
        # THE PATTERN IS THE LAW, THE CLICK ITS CAUSE: no click on a thing
        # before this one (the game may hold a selection of its own) — any
        # kind two of whose things are gone and one new is read
        from collections import Counter
        kinds = [k for k in Counter((s.body, s.size) for s in before if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags) if k[1] >= 2]
    for kind in kinds:
        P0 = [_centre(s) for s in before if s.sid < 1000 and (s.body, s.size) == kind and "hud" not in s.tags]
        P1 = [_centre(s) for s in after if s.sid < 1000 and (s.body, s.size) == kind and "hud" not in s.tags]
        gone = [p for p in P0 if not any(_near(p, q) for q in P1)]
        new = [p for p in P1 if not any(_near(p, q) for q in P0)]
        if len(gone) != 2 or len(new) != 1:
            continue
        dst = new[0]
        # A is the gone thing the other (B) lies between it and the landing
        found = None
        for a0 in gone:
            B = next(g for g in gone if g is not a0)
            v_b = (B[0] - a0[0], B[1] - a0[1])
            v_d = (dst[0] - a0[0], dst[1] - a0[1])
            if _near(v_d, (2 * v_b[0], 2 * v_b[1]), 1.5):
                found = (a0, B); break
        if found is None:
            continue
        a0, B = found
        if under and not any(_near(_centre(min(under, key=lambda s: s.size)), g) for g in gone):
            continue
        A_size = kind[1]
        v_b = (B[0] - a0[0], B[1] - a0[1])
        v_d = (dst[0] - a0[0], dst[1] - a0[1])
        tol = max(2.5, A_size ** 0.5)
        if not (_near(dst, click_now, tol) or _near(a0, click_now, tol) or _near(B, click_now, tol)):
            continue                    # (the click was on the leaper, the leapt-over or the landing: one of the three caused it)
        if (abs(v_b[0]) > 1) == (abs(v_b[1]) > 1):
            continue                    # (a leap is along one axis)
        pitch = int(round(max(abs(v_b[0]), abs(v_b[1]))))
        if pitch < 2:
            continue
        return {"kind": kind, "pitch": pitch, "from": a0, "over": B, "to": dst}
    return None


def lattice(things, kind, pitch, ref):
    """The board as lattice cells: the places of the kind's things and of
    the like tiles they stand among (things of one kind, many alike, at
    the pitch), as integer coordinates from ref."""
    from collections import Counter
    n_kind = Counter((s.body, s.size, s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]) for s in things
                     if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags)
    tile_kinds = {k for k, n in n_kind.items() if n >= 6 and (k[0], k[1]) != kind}
    cells = set(); pegs = set()

    def to_cell(p):
        y = (p[0] - ref[0]) / pitch; x = (p[1] - ref[1]) / pitch
        if abs(y - round(y)) > 0.25 or abs(x - round(x)) > 0.25:
            return None
        return (int(round(y)), int(round(x)))
    for s in things:
        if s.sid >= 1000 or "hud" in s.tags or "terrain" in s.tags:
            continue
        k4 = (s.body, s.size, s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1])
        c = to_cell(s.centre)
        if c is None:
            continue
        if (s.body, s.size) == kind:
            pegs.add(c); cells.add(c)
        elif k4 in tile_kinds:
            cells.add(c)
    return cells, pegs


def leaps_from(board, pegs):
    out = []
    for p in pegs:
        for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            over = (p[0] + d[0], p[1] + d[1])
            dst = (p[0] + 2 * d[0], p[1] + 2 * d[1])
            if over in pegs and dst in board and dst not in pegs:
                out.append((p, over, dst))
    return out


def plan_leaps(board, pegs, node_cap=400000):
    """The fewest of the kind left, by leaps: [(src cell, dst cell)]."""
    board = frozenset(board)
    seen = set()
    best = {"left": len(pegs), "path": []}
    sys.setrecursionlimit(10000)

    def dfs(state, path):
        if len(state) < best["left"]:
            best["left"] = len(state); best["path"] = path
        if len(state) == 1:
            return path
        if state in seen or len(seen) > node_cap:
            return None
        seen.add(state)
        for src, over, dst in leaps_from(board, state):
            ns = frozenset((state - {src, over}) | {dst})
            r = dfs(ns, path + [(src, dst)])
            if r is not None:
                return r
        return None

    full = dfs(frozenset(pegs), [])
    return full if full is not None else best["path"]
