"""Multi-body planning: several bodies respond to the SAME input with
DIFFERENT (typically mirrored) motions, and the goal archetype is to merge
them (m0r0 engine, ground truth: pieces landing on the same cell fuse;
level completes when every piece is fused).

Detection is model-free: any >=2 controlled, non-autonomous tracks whose
per-action direction maps are established (>=2 votes) and not identical.
Planning is joint-state A*: one input moves every body by its own vector;
a body blocked by a wall/bounds simply stays (that asymmetry is the whole
puzzle — walls let you change the bodies' relative position).
"""
from heapq import heappush, heappop


def detect_bodies(tracker, min_actions=2):
    """[(track_id, {action: unit_dir})] for controlled bodies."""
    out = []
    for tid, tr in tracker.tracks.items():
        if tr.missed or tr.is_autonomous or tr.is_oscillating:
            continue
        amap = {}
        for a, ctr in tr.action_moves.items():
            if a in ("ACTION6", "RESET") or not ctr:
                continue
            # (0,0) entries are wall-blocked attempts: they say nothing
            # about the action's DIRECTION, only about the map — dominance
            # is judged among the nonzero displacements alone
            nz = {d: n for d, n in ctr.items() if d != (0, 0)}
            if not nz:
                continue
            d, n = max(nz.items(), key=lambda kv: kv[1])
            if n >= 2 and n >= 0.7 * sum(nz.values()):
                amap[a] = d
        if len(amap) >= min_actions:
            out.append((tid, amap))
    if len(out) < 2:
        return None
    # at least one action must move two bodies differently (else it's just
    # one body seen twice / a follower)
    differ = False
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            common = set(out[i][1]) & set(out[j][1])
            if any(out[i][1][a] != out[j][1][a] for a in common):
                differ = True
    return out if differ else None


def plan_merge(positions, amaps, step, passable, max_nodes=30000, tol=None):
    """A* to bring all bodies together.  positions: [(r, c)];
    amaps: [{action: unit_dir}]; step: px per move; passable(r, c) -> bool.
    Goal: every pairwise distance <= tol (default: one step — the engine's
    adjacent-swap rule finishes the merge).  Returns [action, ...] or None."""
    if tol is None:
        tol = step
    n = len(positions)
    actions = sorted(set().union(*[set(m) for m in amaps]))

    def pair_sum(st):
        tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                tot += (abs(st[i][0] - st[j][0]) + abs(st[i][1] - st[j][1]))
        return tot

    def done(st):
        for i in range(n):
            for j in range(i + 1, n):
                if abs(st[i][0] - st[j][0]) + abs(st[i][1] - st[j][1]) > tol:
                    return False
        return True

    start = tuple(tuple(p) for p in positions)
    if done(start):
        return []
    openq = [(pair_sum(start) // max(step, 1), 0, start, [])]
    seen = {start: 0}
    nodes = 0
    while openq and nodes < max_nodes:
        nodes += 1
        f, g, st, path = heappop(openq)
        for a in actions:
            nxt = []
            moved_any = False
            for i in range(n):
                d = amaps[i].get(a)
                if d is None:
                    nxt.append(st[i])
                    continue
                nr, nc = st[i][0] + d[0] * step, st[i][1] + d[1] * step
                if passable(nr, nc):
                    nxt.append((nr, nc))
                    moved_any = True
                else:
                    nxt.append(st[i])
            if not moved_any:
                continue
            tst = tuple(nxt)
            ng = g + 1
            if seen.get(tst, 1 << 30) <= ng or ng > 80:
                continue
            seen[tst] = ng
            npath = path + [a]
            if done(tst):
                return npath
            heappush(openq, (ng + pair_sum(tst) // max(step, 1), ng, tst,
                             npath))
    return None
