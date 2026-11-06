"""Static scene analysis: the LOOK phase.

Applies universal priors (given, never learned — the 'mathematics' of grid
worlds) to a single frame, producing relational facts and goal hypotheses
BEFORE any action is spent:

  - panel clustering: objects group into visual panels
  - congruence: two panels with identical content = template/copy
  - near-congruence: panels differing in k cells = 'make X match Y'
    (canvas = the odd one out; template = the repeated/majority pattern)

Output: MatchTask hypotheses the agent can test by acting on diff cells.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def cluster_objects(objs, gap=4):
    """Group objects whose bboxes are within `gap` cells into panels."""
    n = len(objs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def near(a, b):
        ar0, ac0, ar1, ac1 = a.bbox
        br0, bc0, br1, bc1 = b.bbox
        return not (ar0 > br1 + gap or br0 > ar1 + gap
                    or ac0 > bc1 + gap or bc0 > ac1 + gap)

    for i in range(n):
        for j in range(i + 1, n):
            if near(objs[i], objs[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(objs[i])
    panels = []
    for members in groups.values():
        r0 = min(o.bbox[0] for o in members)
        c0 = min(o.bbox[1] for o in members)
        r1 = max(o.bbox[2] for o in members)
        c1 = max(o.bbox[3] for o in members)
        panels.append(((r0, c0, r1, c1), members))
    return panels


def trim_uniform_border(crop):
    """Strip uniform frame rows/cols (panel borders); return (sub, r_off, c_off)."""
    r0, c0 = 0, 0
    r1, c1 = crop.shape
    def uni(line):
        return (line == line[0]).all()
    while r1 - r0 > 2 and uni(crop[r0]):
        r0 += 1
    while r1 - r0 > 2 and uni(crop[r1 - 1]):
        r1 -= 1
    while c1 - c0 > 2 and uni(crop[r0:r1, c0]):
        c0 += 1
    while c1 - c0 > 2 and uni(crop[r0:r1, c1 - 1]):
        c1 -= 1
    return crop[r0:r1, c0:c1], r0, c0


@dataclass
class MatchTask:
    """'Make the canvas match the template.'"""
    canvas_bbox: tuple            # logical coords of canvas panel (untrimmed)
    diff_cells: list              # [(logical_r, logical_c, desired_color)]
    agreement: float
    template_bbox: tuple
    support: int = 1              # how many identical template copies exist

    @property
    def confidence(self):
        return self.agreement * (1 + 0.3 * (self.support - 1))


def abstract_block_grid(members, bg):
    """Coarse-graining prior: a cluster of same-size uniform blocks on a
    regular lattice abstracts to a small matrix (one block = one cell).
    Returns (matrix, positions) or None; positions[r][c] = logical centroid."""
    from collections import Counter as C
    if len(members) < 4:
        return None
    area = C()
    for o in members:
        area[o.size] += o.size          # weight by total covered area:
    modal_size, _ = area.most_common(1)[0]  # 9 marker specks must not
    blocks = [o for o in members if o.size == modal_size]  # outvote 8 cells
    if len(blocks) < 4:
        return None
    rows = sorted(set(o.bbox[0] for o in blocks))
    cols = sorted(set(o.bbox[1] for o in blocks))
    if len(rows) < 2 and len(cols) < 2:
        return None
    if len(rows) * len(cols) < len(blocks) * 0.8:
        return None
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: i for i, c in enumerate(cols)}
    mat = np.full((len(rows), len(cols)), bg, dtype=int)
    # every lattice slot gets a position — EMPTY cells included (an
    # unfilled canvas cell must still be clickable)
    bh = blocks[0].bbox[2] - blocks[0].bbox[0]
    bw = blocks[0].bbox[3] - blocks[0].bbox[1]
    pos = {}
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            pos[(i, j)] = (int(r + bh / 2), int(c + bw / 2))
    for o in blocks:
        i, j = ri[o.bbox[0]], ci[o.bbox[1]]
        mat[i, j] = o.color
        pos[(i, j)] = (int(round(o.centroid[0])), int(round(o.centroid[1])))
    return mat, pos


def find_block_match_tasks(objs, bg):
    """Match tasks between grids-of-blocks (one block = one super-cell)."""
    panels = cluster_objects([o for o in objs if o.size >= 3])
    grids = []
    for bbox, members in panels:
        g = abstract_block_grid(members, bg)
        if g is not None:
            grids.append({"bbox": bbox, "mat": g[0], "pos": g[1]})
    sig = Counter()
    for g in grids:
        sig[(g["mat"].shape, g["mat"].tobytes())] += 1
    tasks = []
    def palette_jaccard(x, y):
        cx = set(np.unique(x).tolist())
        cy = set(np.unique(y).tolist())
        return len(cx & cy) / max(len(cx | cy), 1)

    for i in range(len(grids)):
        for j in range(len(grids)):
            if i == j:
                continue
            a, b = grids[i], grids[j]          # a = canvas, b = template
            if a["mat"].shape != b["mat"].shape:
                continue
            # palette-compatibility prior: a template whose colors the
            # canvas cannot display is not a literal template. But it can be
            # a SYMBOLIC one (a mini-map legend): if both it and the shared
            # group palette have exactly 2 colors, enumerate both bijections
            # as separate low-confidence hypotheses.
            symbolic_maps = []
            if palette_jaccard(a["mat"], b["mat"]) < 0.34:
                tcolors = sorted(int(v) for v in np.unique(b["mat"]))
                group_pal = set()
                for g in grids:
                    if g is b or g["mat"].shape != b["mat"].shape:
                        continue
                    group_pal |= set(int(v) for v in np.unique(g["mat"]))
                group_pal = sorted(group_pal - set(tcolors))
                if len(tcolors) == 2 and len(group_pal) == 2:
                    symbolic_maps = [
                        {tcolors[0]: group_pal[0], tcolors[1]: group_pal[1]},
                        {tcolors[0]: group_pal[1], tcolors[1]: group_pal[0]},
                    ]
                else:
                    continue
            for smap in symbolic_maps:
                mapped_b = np.vectorize(lambda v: smap[int(v)])(b["mat"])
                eqs = a["mat"] == mapped_b
                agrs = float(eqs.mean())
                if agrs >= 1.0:
                    continue
                sdiffs = []
                for (r, c) in zip(*np.where(~eqs)):
                    p = a["pos"].get((int(r), int(c)))
                    if p is not None:
                        sdiffs.append((p[0], p[1], int(mapped_b[r, c])))
                if sdiffs:
                    tasks.append(MatchTask(
                        canvas_bbox=a["bbox"], diff_cells=sdiffs,
                        agreement=max(agrs, 0.31) * 0.9,
                        template_bbox=(b["bbox"], tuple(sorted(smap.items()))),
                        support=1))
            if symbolic_maps:
                continue
            # matching under two hypotheses, best wins:
            #  1. identity — shared palette (an unstarted canvas mostly
            #     disagrees with its template; bijections reject it)
            #  2. color bijection — template palette differs from canvas's
            candidates = [b["mat"]]
            mapping = {}
            used = set()
            ok = True
            for tc in np.unique(b["mat"]):
                vals, counts = np.unique(a["mat"][b["mat"] == tc],
                                         return_counts=True)
                top = int(vals[np.argmax(counts)])
                if top in used:
                    ok = False
                    break
                mapping[int(tc)] = top
                used.add(top)
            if ok and any(k != v for k, v in mapping.items()):
                candidates.append(
                    np.vectorize(lambda v: mapping[int(v)])(b["mat"]))
            best_eq, best_agr, mapped = None, -1.0, None
            for cand in candidates:
                eq_c = a["mat"] == cand
                if float(eq_c.mean()) > best_agr:
                    best_eq, best_agr, mapped = eq_c, float(eq_c.mean()), cand
            eq, agr = best_eq, best_agr
            if agr >= 1.0 or agr < 0.3:
                continue
            diffs = []
            for (r, c) in zip(*np.where(~eq)):
                p = a["pos"].get((int(r), int(c)))
                if p is not None:
                    diffs.append((p[0], p[1], int(mapped[r, c])))
            if not diffs:
                continue
            tasks.append(MatchTask(
                canvas_bbox=a["bbox"], diff_cells=diffs, agreement=agr,
                template_bbox=b["bbox"],
                support=sig[(b["mat"].shape, b["mat"].tobytes())]))
    # analogy prior (A:B :: C:D): four same-shape grids in reading order —
    # the top pair demonstrates an edit; the target for the last grid (the
    # canvas) is that edit applied to the third: where(A==B, C, B)
    by_shape_a = {}
    for g in grids:
        by_shape_a.setdefault(g["mat"].shape, []).append(g)
    for shape, group in by_shape_a.items():
        if len(group) < 4:
            continue
        ordered = sorted(group, key=lambda g: (g["bbox"][0] // 8, g["bbox"][1]))
        A, B, C, D = ordered[0], ordered[1], ordered[2], ordered[-1]
        pj = lambda x, y: len(set(np.unique(x).tolist()) & set(np.unique(y).tolist())) / \
            max(len(set(np.unique(x).tolist()) | set(np.unique(y).tolist())), 1)
        if min(pj(A["mat"], B["mat"]), pj(B["mat"], C["mat"])) >= 0.34:
            target = np.where(A["mat"] == B["mat"], C["mat"], B["mat"])
            eq = D["mat"] == target
            if float(eq.mean()) < 1.0:
                diffs = []
                for (r, c) in zip(*np.where(~eq)):
                    p = D["pos"].get((int(r), int(c)))
                    if p is not None:
                        diffs.append((p[0], p[1], int(target[r, c])))
                if diffs:
                    tasks.append(MatchTask(
                        canvas_bbox=D["bbox"], diff_cells=diffs,
                        agreement=0.99, template_bbox=("analogy",),
                        support=3))

    # consensus prior: with >=3 same-shape grids, the target for any canvas
    # is the CELLWISE MAJORITY of the others (repeated information = signal)
    from collections import defaultdict as dd
    by_shape = dd(list)
    for g in grids:
        by_shape[g["mat"].shape].append(g)
    def _pj(x, y):
        cx = set(np.unique(x).tolist())
        cy = set(np.unique(y).tolist())
        return len(cx & cy) / max(len(cx | cy), 1)

    for shape, group in by_shape.items():
        if len(group) < 3:
            continue
        for canvas in group:
            others = [g["mat"] for g in group if g is not canvas
                      and _pj(g["mat"], canvas["mat"]) >= 0.34]
            if len(others) < 2:
                continue
            stack = np.stack(others)
            h, w = shape
            target = np.zeros(shape, dtype=int)
            for r in range(h):
                for c in range(w):
                    vals, counts = np.unique(stack[:, r, c], return_counts=True)
                    target[r, c] = int(vals[np.argmax(counts)])
            eq = canvas["mat"] == target
            agr = float(eq.mean())
            if agr >= 1.0 or agr < 0.2:
                continue
            diffs = []
            for (r, c) in zip(*np.where(~eq)):
                p = canvas["pos"].get((int(r), int(c)))
                if p is not None:
                    diffs.append((p[0], p[1], int(target[r, c])))
            if diffs:
                tasks.append(MatchTask(
                    canvas_bbox=canvas["bbox"], diff_cells=diffs,
                    agreement=agr, template_bbox=None,
                    support=len(others)))
    tasks.sort(key=lambda t: (-t.confidence, len(t.diff_cells)))
    return tasks


def find_match_tasks(logical, objs, bg):
    """Detect template/canvas panel pairs and their differing cells."""
    panels = cluster_objects([o for o in objs if o.size >= 3])
    crops = []
    for bbox, members in panels:
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        crop = logical[r0:r1 + 1, c0:c1 + 1]
        sub, ro, co = trim_uniform_border(crop)
        if sub.size < 4:
            continue
        crops.append({"bbox": bbox, "sub": sub,
                      "off": (bbox[0] + ro, bbox[1] + co)})
    tasks = []
    # count identical panels (template support)
    sig = Counter()
    for c in crops:
        sig[(c["sub"].shape, c["sub"].tobytes())] += 1
    for i in range(len(crops)):
        for j in range(len(crops)):
            if i == j:
                continue
            a, b = crops[i], crops[j]          # a = canvas, b = template
            if a["sub"].shape != b["sub"].shape:
                continue
            eq = (a["sub"] == b["sub"])
            agr = float(eq.mean())
            if agr >= 1.0 or agr < 0.4:
                continue
            support = sig[(b["sub"].shape, b["sub"].tobytes())]
            diffs = []
            ar, ac = a["off"]
            for (r, c) in zip(*np.where(~eq)):
                diffs.append((int(ar + r), int(ac + c),
                              int(b["sub"][r, c])))
            tasks.append(MatchTask(
                canvas_bbox=a["bbox"], diff_cells=diffs, agreement=agr,
                template_bbox=b["bbox"], support=support))
    # prefer high-support templates and high agreement, small diffs first
    tasks.sort(key=lambda t: (-t.confidence, len(t.diff_cells)))
    return tasks


def playfield_symmetry_defects(field, bg):
    """Mirror-symmetry prior: if the field is one cell-flip away from being
    left-right or top-bottom symmetric, those defect cells are goal
    candidates ('restore symmetry')."""
    out = []
    h, w = field.shape
    lr = field[:, ::-1]
    mism = field != lr
    frac = float(mism.mean())
    if 0 < frac <= 0.1:
        for (r, c) in zip(*np.where(mism)):
            out.append((int(r), int(c), int(lr[r, c])))
    return out
