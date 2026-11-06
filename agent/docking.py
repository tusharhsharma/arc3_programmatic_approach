"""Dock-and-orient: fit the avatar (as a RIGID GROUP) into a socket group
in the right ORIENTATION.

Triggered when the agent has discovered a rotate action (an action whose
effect on the avatar's own cells is a dihedral op). The goal hypothesis is
generic: some static group has a cavity/port the avatar group complements;
docking there at the right pose is what the level wants (plug/socket,
key/lock, piece/slot).

Real pieces are COMPOUND objects (a plug body plus its prongs), so cells
carry their color: shapes are frozensets of (r, c, color). Matching rewards
same-color contact across the two groups — prongs mate with notches.

Pure geometry here; the mapper owns state and execution.
"""


def cnorm(cells):
    """Normalize a colored cell set to its bbox origin."""
    r0 = min(r for r, _, _ in cells)
    c0 = min(c for _, c, _ in cells)
    return frozenset((r - r0, c - c0, col) for r, c, col in cells)


def crot90(cells):
    h = max(r for r, _, _ in cells) + 1
    return cnorm(frozenset((c, h - 1 - r, col) for r, c, col in cells))


def crot_k(cells, k):
    out = cnorm(cells)
    for _ in range(k % 4):
        out = crot90(out)
    return out


def orientation_of(base, cur):
    """j such that crot_k(base, j) == cnorm(cur), else None."""
    n = cnorm(cur)
    for j in range(4):
        if crot_k(base, j) == n:
            return j
    # POSITIONS-ONLY fallback: port cells RECOLOR when they connect (that
    # recolor IS the success signal), so an exact colored match fails at
    # the very moment of docking. Geometry alone still identifies the
    # orientation; the success check reads the colors separately.
    npos = {(r, c) for r, c, _ in n}
    for j in range(4):
        if {(r, c) for r, c, _ in crot_k(base, j)} == npos:
            return j
    return None


def group_cells(seed_obj, objs, wcells, max_gap=1, max_ratio=6.0):
    """Rigid group: the seed object plus attached parts (cell-adjacent
    within max_gap). Colored world cells of the whole group."""
    seed_w = wcells(seed_obj)
    group = [seed_obj]
    cells = {(r, c) for r, c, _ in seed_w}
    colored = set(seed_w)
    added = True
    while added:
        added = False
        for o in objs:
            if o in group or o.size > max_ratio * max(seed_obj.size, 1):
                continue
            w = wcells(o)
            if any(abs(r - r2) <= max_gap and abs(c - c2) <= max_gap
                   for r, c, _ in w for (r2, c2) in cells):
                group.append(o)
                cells |= {(r, c) for r, c, _ in w}
                colored |= w
                added = True
    return frozenset(colored), group


def port_colors(av_colored, sock_colored):
    """Connector colors: present in BOTH groups AND a minority color of
    the avatar group (ports are small features, not bodies)."""
    from collections import Counter
    counts = Counter(col for _, _, col in av_colored)
    a = {col for col, n in counts.items() if n <= 0.3 * len(av_colored)}
    s = {col for _, _, col in sock_colored}
    return a & s


def best_poses(av_colored, sock_colored, is_bg, refuted, top=8, lattice=None, bounds=None):
    """Rank (k, topleft) placements of the rotated avatar group against the
    socket group. Ground truth (cn04 engine): the win condition is EXACT
    same-cell coincidence of same-color PORT cells across two sprites;
    bodies overlap freely (no collision). Score = port coincidences, with a
    perfect-cover bonus when every avatar port lands on a socket port."""
    base = cnorm(av_colored)
    ports = port_colors(av_colored, sock_colored)
    if not ports:
        return []
    sock_ports = {}
    for r, c, col in sock_colored:
        if col in ports:
            sock_ports.setdefault(col, set()).add((r, c))
    rs = [r for r, c, _ in sock_colored]
    cs = [c for _, c, _ in sock_colored]
    sr0, sr1, sc0, sc1 = min(rs), max(rs), min(cs), max(cs)
    poses = []
    for k in range(4):
        shape = crot_k(base, k)
        av_ports = [(r, c, col) for r, c, col in shape if col in ports]
        if not av_ports:
            continue
        h = max(r for r, _, _ in shape) + 1
        w = max(c for _, c, _ in shape) + 1
        for tr in range(sr0 - h - 1, sr1 + 2):
            for tc in range(sc0 - w - 1, sc1 + 2):
                if bounds is not None:
                    fh, fw = bounds
                    if tr < 0 or tc < 0 or tr + h > fh or tc + w > fw:
                        continue    # pose sticks out of the arena: unreachable
                if lattice is not None:
                    ar0, ac0, step = lattice
                    # movement happens in engine-grid steps: a pose off the
                    # avatar's step-lattice is physically unreachable and
                    # must not be proposed (phantom 1-px near-misses)
                    if step > 1 and ((tr - ar0) % step or (tc - ac0) % step):
                        continue
                hit = sum(1 for r, c, col in av_ports
                          if (tr + r, tc + c) in sock_ports.get(col, ()))
                if hit == 0:
                    continue
                score = 10 * hit + (50 if hit == len(av_ports) else 0)
                # REFUTATIONS ATTACH TO PHYSICAL PLACEMENTS: keyed by the
                # placed piece's world-cell footprint, not (k, tl) — after
                # a rotate press the base renormalizes and the same
                # physical pose reappears under a fresh (k, tl), so a
                # representation-dependent key forgets every refutation
                # (observed cn04 L2: the same placement re-committed
                # under four different keys)
                fp = frozenset((tr + r, tc + c) for r, c, _ in shape)
                if fp not in refuted:
                    poses.append((score, k, (tr, tc), fp))
    # ports of different roles can render identically (cn04 renders both
    # port colors the same), so several placements tie at full cover and
    # only one wins: NEVER truncate away full-cover candidates. Tie-break
    # by distance from the avatar's current position (cheap attempts first).
    a_r0 = min(r for r, _, _ in av_colored)
    a_c0 = min(c for _, c, _ in av_colored)
    poses.sort(key=lambda p: (-p[0],
                              abs(p[2][0] - a_r0) + abs(p[2][1] - a_c0)))
    full = [p for p in poses if p[0] >= 50]
    return full if full else poses[:top]


def pose_target(base, k, topleft):
    """World centroid cell of the rotated avatar group placed at topleft."""
    shape = crot_k(base, k)
    rr = sum(r for r, _, _ in shape) / len(shape)
    cc = sum(c for _, c, _ in shape) / len(shape)
    return (round(topleft[0] + rr), round(topleft[1] + cc))
