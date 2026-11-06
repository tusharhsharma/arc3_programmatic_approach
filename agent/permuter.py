"""Permutation-operator puzzles (lp85 engine, ground truth): the only
effective action is clicking a BUTTON, and each button applies a fixed
rearrangement (a cyclic shift) of the movable pieces over a loop of cells.
Win = every piece parked on a marker cell of its kind.

The laws of cheap learning apply in their purest form here:
  - ONE click per button reveals that button's entire operator
    (verification by use: the solution's own clicks re-confirm it);
  - a button pair's twin is usually the inverse — learned the same way;
  - once operators are known, the puzzle is solved IN THE HEAD: graph
    search over configurations costs zero game actions.

Pure logic here; the mapper owns observation and execution.
"""
from collections import deque


def op_from_moves(moves):
    """moves: [(from_cell, to_cell)] observed after one click ->
    {from: to} operator, or None if degenerate."""
    op = {}
    for a, b in moves:
        if a == b or a in op:
            return None
        op[a] = b
    # ONE clean pair is valid evidence (Law 1) — the caller decides how
    # many pairs it takes to BELIEVE a new button (two, to reject stray
    # coincidences); growing a known button's operator takes just one.
    return op if len(op) >= 1 else None


def augment_inverses(ops):
    """Button pairs are commonly a loop's two directions. DETECT the
    inverse relation instead of assuming it: k2 inverts k1 when their
    maps agree as inverses on >=2 witnessed cells and contradict on none
    (two different loops share no cells and would pass vacuously — the
    witness requirement blocks that). A detected inverse donates its
    mirror mappings to the cells the partner has not seen yet: half the
    operator comes free (Law 3).  ops: {key: {a: b}} -> augmented copy."""
    out = {k: dict(v) for k, v in ops.items()}
    for k1, o1 in ops.items():
        for k2, o2 in ops.items():
            if k1 == k2:
                continue
            inv2 = {b: a for a, b in o2.items()}
            wit = sum(1 for a, b in o1.items() if inv2.get(a) == b)
            contra = any(inv2.get(a, b) != b for a, b in o1.items())
            if contra or wit < 2:
                continue
            for a, b in inv2.items():
                out[k1].setdefault(a, b)
    return out


def close_ring(op, thr):
    """Infer the FULL cyclic operator from a partial one. The op's cells
    (keys + values) lie on a track whose consecutive cells are one step
    (<= thr) apart. Two track shapes close into a cycle:
      - a RING: every cell has exactly two step-neighbors;
      - a LINE with WRAPAROUND: two endpoint cells with one neighbor —
        the shift's wrap edge (end -> start) is a long jump that local
        move observation can never certify, but the topology implies it.
    When every known mapping follows one direction around the track, the
    whole rotation is determined — the operator's FORM is inferred and
    its first planned use verifies it (Law 1). A cell set with interior
    gaps splits into >2 endpoints and the inference safely declines."""
    cells = set(op) | set(op.values())
    if len(cells) < 4 or len(op) >= len(cells):
        return op
    nb = {}
    ends = []
    for c in cells:
        ns = [d for d in cells if d != c
              and abs(c[0] - d[0]) + abs(c[1] - d[1]) <= thr]
        if len(ns) == 1:
            ends.append(c)
        elif len(ns) != 2:
            return op
        nb[c] = ns
    if len(ends) not in (0, 2):
        return op
    start = ends[0] if ends else next(iter(cells))
    seq = [start, nb[start][0]]
    while True:
        nxt = [d for d in nb[seq[-1]] if d != seq[-2]]
        if len(seq) > len(cells):
            return op
        if not nxt or nxt[0] == start:
            break
        seq.append(nxt[0])
    if len(seq) != len(cells):
        return op
    idx = {c: i for i, c in enumerate(seq)}
    n = len(seq)
    fwd = sum(1 for a, b in op.items() if idx[b] == (idx[a] + 1) % n)
    bwd = sum(1 for a, b in op.items() if idx[b] == (idx[a] - 1) % n)
    if fwd and not bwd:
        return {seq[i]: seq[(i + 1) % n] for i in range(n)}
    if bwd and not fwd:
        return {seq[i]: seq[(i - 1) % n] for i in range(n)}
    return op


def perm_solve(pieces, ops, goal_cells_by_class, max_depth=64,
               max_nodes=250000):
    """Search over configurations — in the QUOTIENT space.

    THE CONCEPTUAL KEY: operators are pure cell-permutations, so pieces
    never interact — a piece's trajectory depends only on its own cell.
    Therefore only the CONSTRAINED pieces (those with goal cells) need to
    be tracked; filler pieces are dynamically irrelevant. The state space
    collapses from |cells|^|pieces| to |cells|^|constrained| — a few
    thousand states for ANY number of loops.

    pieces: {cell: piece_class}; ops: {key: {from: to}};
    goal_cells_by_class: {piece_class: set(cells)}.
    Returns [op_key, ...] or None."""
    tracked = [(cell, cls) for cell, cls in pieces.items()
               if cls in goal_cells_by_class]
    if not tracked:
        return None
    cells0 = tuple(cell for cell, _ in tracked)
    classes = tuple(cls for _, cls in tracked)

    # COVER SEMANTICS under partial observability: goals hidden beneath
    # never-moved pieces are invisible; only the KNOWN goal cells
    # constrain — the win test is that every known goal cell of a class
    # is covered by a piece of that class (a piece parked on its hidden
    # goal is satisfied in place; if we vacate it, the goal reveals and
    # the memory grows).
    def is_goal(cells):
        for cls in set(classes):
            want = goal_cells_by_class[cls]
            have = {c for c, cl in zip(cells, classes) if cl == cls}
            if not want <= have:
                return False
        return True

    if is_goal(cells0):
        return []
    q = deque([(cells0, [])])
    seen = {cells0}
    nodes = 0
    while q and nodes < max_nodes:
        state, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for key, op in ops.items():
            nxt = tuple(op.get(c, c) for c in state)
            if nxt == state or nxt in seen:
                continue
            seen.add(nxt)
            npath = path + [key]
            if is_goal(nxt):
                return npath
            nodes += 1
            q.append((nxt, npath))
    return None


def goal_assignments(piece_classes, marker_cells_by_class):
    """Candidate {piece_class: goal cell set} assignments, most plausible
    first. Ground truth (lp85): a goal marker WEARS THE COLOR of the piece
    class it wants — color-identity mapping first; classes without a
    matching marker are unconstrained filler. Fallback: everything onto
    any marker."""
    out = []
    ident = {}
    for pc in piece_classes:
        col = pc[0] if isinstance(pc, tuple) else pc
        if col in marker_cells_by_class and marker_cells_by_class[col]:
            ident[pc] = set(marker_cells_by_class[col])
    if ident:
        out.append(ident)
    allcells = set().union(*marker_cells_by_class.values()) \
        if marker_cells_by_class else set()
    if allcells:
        out.append({pc: allcells for pc in piece_classes})
    return out
