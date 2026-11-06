"""Arrange-then-commit games (sp80 engine, ground truth): a movable
DEFLECTOR piece, a SOURCE emitting downward (a spout), several TARGET
basins of one class, and a limited-budget COMMIT action (pour) that tests
the arrangement irreversibly.

Ground-truth flow rule generalized: a stream hitting the deflector runs to
its ends and falls just outside them — so the winning placements are the
pure GEOMETRY (free information, Law 3):
    piece spans the source column, and
    (left end - 1) is over one target while (right end + width) is over
    another.
Candidates are enumerated from statics alone; each death refutes one.
"""


def commit_candidates(effect_counts, known_motions, shape_ops):
    """Actions that repeatably change the world but neither move the
    avatar nor transform it: the 'do it' actions (pour, fire, confirm)."""
    out = []
    for name, n in effect_counts.items():
        if n >= 3 and name not in known_motions and name not in shape_ops \
                and name != "ACTION6":
            out.append(name)
    return out


def deflector_placements(piece_w, source_col, targets, bounds_w, refuted):
    """Candidate x positions for the deflector.
    targets: list of (col_lo, col_hi) spans of the basins.
    Winning rule: x-1 lands in one span, x+piece_w in another, and
    source_col within [x, x+piece_w-1]."""
    out = []
    for i in range(len(targets)):
        for j in range(len(targets)):
            if i == j:
                continue
            alo, ahi = targets[i]
            blo, bhi = targets[j]
            for x in range(max(0, alo + 1), min(bounds_w - piece_w, ahi + 1) + 1):
                if not (alo <= x - 1 <= ahi):
                    continue
                if not (blo <= x + piece_w <= bhi):
                    continue
                if not (x <= source_col <= x + piece_w - 1):
                    continue
                if x not in refuted:
                    out.append(x)
    # nearest-to-source first (cheap moves, likely intended)
    out = sorted(set(out), key=lambda x: abs(x + piece_w // 2 - source_col))
    return out
