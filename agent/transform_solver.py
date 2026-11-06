"""TransformSolver: generic ontology for rotate/flip shape-matching games.

Pattern: a static TEMPLATE shape and a manipulable PIECE; actions apply
rigid transforms (rotations/mirrors) to the piece; the level is won by
making the piece congruent with the template (then possibly confirming).

Learning: per action, identify which dihedral op maps the piece's cell-set
before -> after. Planning: BFS over learned ops (depth <= 4) to reach a
configuration congruent with the template (IoU-tolerant, so a dotted or
textured template still matches its solid counterpart).
"""
from collections import defaultdict


def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _dims(cells):
    return (max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1)


def rot90(cells):
    h, _ = _dims(cells)
    return _norm(frozenset((c, h - 1 - r) for r, c in cells))


def rot180(cells):
    return rot90(rot90(cells))


def rot270(cells):
    return rot90(rot180(cells))


def flip_h(cells):
    _, w = _dims(cells)
    return _norm(frozenset((r, w - 1 - c) for r, c in cells))


def flip_v(cells):
    h, _ = _dims(cells)
    return _norm(frozenset((h - 1 - r, c) for r, c in cells))


OPS = {"rot90": rot90, "rot180": rot180, "rot270": rot270,
       "flip_h": flip_h, "flip_v": flip_v}


def iou(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def congruent(a, b, thresh=0.7):
    """a matches b under identity (translation already normalized).
    IoU-tolerant: a textured/dotted template still matches its solid twin."""
    return iou(a, b) >= thresh


class TransformSolver:
    MIN_SIZE = 4          # ignore specks
    MAX_SIZE = 200        # ignore backgrounds

    def __init__(self):
        self.op_votes = defaultdict(lambda: defaultdict(int))  # action -> op -> n
        self.no_change = defaultdict(int)                      # action -> n
        self.static_count = defaultdict(int)   # centroid key -> stable steps
        self.last_shapes = {}                  # centroid key -> norm cells

    def _key(self, o):
        return (round(o.centroid[0] / 2), round(o.centroid[1] / 2))

    def _relevant(self, objs):
        return [o for o in objs if self.MIN_SIZE <= o.size <= self.MAX_SIZE]

    # ---------- learning ----------

    def update(self, action_name, objs_before, objs_after):
        before = {self._key(o): _norm(o.cells) for o in self._relevant(objs_before)}
        after = {self._key(o): _norm(o.cells) for o in self._relevant(objs_after)}
        changed_any = False
        matched_before, matched_after = set(), set()
        for k, shape_a in after.items():
            shape_b = before.get(k)
            if shape_b is None:
                # rotation shifts the bbox: try neighboring keys
                for dk in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1),
                           (1, -1), (-1, 1)):
                    shape_b = before.get((k[0] + dk[0], k[1] + dk[1]))
                    if shape_b is not None:
                        break
            if shape_b is None:
                continue
            if shape_a == shape_b:
                self.static_count[k] += 1
                matched_after.add(k)
                matched_before.add(k)
            else:
                changed_any = True
                self.static_count[k] = 0
                matched_after.add(k)
                matched_before.add(k)
                for op_name, op in OPS.items():
                    # engine transforms can differ by a pixel: tolerate
                    if iou(op(shape_b), shape_a) >= 0.9:
                        self.op_votes[action_name][op_name] += 1
                        break
            self.last_shapes[k] = shape_a
        # global pairing: a rotation can move the bbox far from its old
        # key — if exactly one shape (dis)appeared on each side, pair them
        lb = [v for k, v in before.items() if k not in matched_before]
        la = [v for k, v in after.items() if k not in matched_after]
        if len(lb) == 1 and len(la) == 1 and lb[0] != la[0]:
            changed_any = True
            for op_name, op in OPS.items():
                if iou(op(lb[0]), la[0]) >= 0.9:
                    self.op_votes[action_name][op_name] += 1
                    break
        if not changed_any:
            self.no_change[action_name] += 1

    def action_ops(self):
        """{action: op_name} for actions with a consistent learned op."""
        out = {}
        for a, votes in self.op_votes.items():
            op, n = max(votes.items(), key=lambda kv: kv[1])
            total = sum(votes.values())
            # move-budget games can't afford double confirmation: accept a
            # single unanimous observation, or a >=60% majority of several
            if (n == total and n >= 1) or (n >= 2 and n >= 0.6 * total):
                out[a] = op
        return out

    def commit_candidates(self):
        """Actions that never transform anything: likely confirm/submit."""
        transforming = set(self.action_ops())
        return [a for a, n in self.no_change.items()
                if n >= 2 and a not in transforming]

    # ---------- solving ----------

    def solve(self, objs):
        """If a (piece, template) pair exists where a <=4-op sequence makes
        them congruent, return the action list; [] if already congruent;
        None when the pattern doesn't apply."""
        ops = self.action_ops()
        if not ops:
            return None
        rel = self._relevant(objs)
        if len(rel) < 2:
            return None
        # template = shape stable for several steps; piece = recently changed
        stat = sorted(rel, key=lambda o: -self.static_count.get(self._key(o), 0))
        templates = [o for o in stat
                     if self.static_count.get(self._key(o), 0) >= 3]
        pieces = [o for o in rel
                  if self.static_count.get(self._key(o), 0) < 3]
        if not templates or not pieces:
            return None
        for piece in pieces:
            p = _norm(piece.cells)
            for tmpl in templates:
                t = _norm(tmpl.cells)
                if abs(piece.size - tmpl.size) > 0.5 * max(piece.size, tmpl.size):
                    continue
                if congruent(p, t):
                    return []          # already matched: caller should commit
                # BFS over learned ops, depth <= 4
                frontier = [(p, [])]
                seen = {p}
                for _ in range(4):
                    nxt = []
                    for shape, seq in frontier:
                        for act, op_name in ops.items():
                            s2 = OPS[op_name](shape)
                            if s2 in seen:
                                continue
                            seq2 = seq + [act]
                            if congruent(s2, t):
                                return seq2
                            seen.add(s2)
                            nxt.append((s2, seq2))
                    frontier = nxt
        return None
