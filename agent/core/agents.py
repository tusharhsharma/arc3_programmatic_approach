"""THE PARALLEL RECORD — what a thing that travels on its own is doing, why,
and what it will do next.

Tushar (2026-10-24): "a single input can have 3 different resulting actions
— identifying that is a basic acknowledgement; identifying the pattern is
intelligence: anything and everything that happens, gauging why it
happened and what would be next."  An agent's tick is an input of its own.
This record keeps, per agent: where it went each press (its STEPS), what it
kept closing on (its WANT — the kind of thing it goes toward), what it did
on arriving (its ACTS: the thing beside it vanished, went with it, went
away from it), and where what it moved came to rest (the fiddle's `rests`).
From the record: a SIGN against the level's goal (what it moves goes toward
the goal's place — an ally; away — a foe), and a PREDICTION of its next
place (one step of its own stride toward the nearest thing of its want,
along the free way), which the next frame confirms or breaks (`misses`).
Nothing here names a game.  Hand-chosen: a want needs three approaches and
more approaches than retreats; a prediction broken three times in a row
puts the want in doubt (re-read from the record)."""
from collections import Counter, defaultdict


def _centre(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _adjacent(a, b):
    """Two boxes touching along an edge (or overlapping)."""
    return not (a[0] > b[2] + 1 or b[0] > a[2] + 1 or a[1] > b[3] + 1 or b[1] > a[3] + 1)


class AgentTrack:
    def __init__(self, sid, kind):
        self.sid = sid
        self.kind = kind
        self.steps = []                  # vectors it moved by, press after press (0,0 when it stood)
        self.stride = None               # the length of its step
        self.approach = defaultdict(Counter)   # kind -> {"toward": n, "away": n}
        self.acts = Counter()            # on arriving beside a thing of its want: "took" / "carried" / "pushed" / "nothing"
        self.sign = Counter()            # +1 a thing it moved went toward the goal's place; -1 away
        self.beside = None               # (sid, bbox) of the want-thing it stood beside last press
        self.predicted = None            # the bbox the record said it would stand at
        self.misses = 0                  # predictions broken in a row
        self.hits = 0
        self.doubted = 0                 # wants withdrawn after missed predictions

    def want(self):
        """The kind it goes toward: most approaches, more than retreats, at least three."""
        best = None
        for kd, c in self.approach.items():
            t_, a_ = c["toward"], c["away"]
            if t_ >= 3 and t_ > a_ and (best is None or t_ - a_ > best[1]):
                best = (kd, t_ - a_)
        return best[0] if best else None

    def stance(self):
        """ally / foe / unknown, by what it did to the things it moved."""
        s = sum(self.sign.values())
        n = sum(abs(v) for v in self.sign.values())
        if n < 2:
            return "unknown"
        return "ally" if s > 0 else ("foe" if s < 0 else "unknown")

    def describe(self):
        w = self.want()
        acts = ", ".join(f"{k} {n}x" for k, n in self.acts.most_common(3))
        return (f"kind {self.kind}: stride {self.stride}, goes toward {('the nearest of kind %s' % w) if w is not None else 'nothing yet'}"
                f"{(' and on arriving: ' + acts) if acts else ''}; {self.stance()} ({dict(self.sign)}); predictions {self.hits} held, {self.misses} broken in a row")


class AgentRecord:
    """All agents of a level: fed every press from the fiddle's before/after."""

    def __init__(self):
        self.tracks = {}                 # sid -> AgentTrack
        self.goal_place = None           # the place the level's reading names (a bbox), when known

    def track(self, sid, kind):
        t = self.tracks.get(sid)
        if t is None:
            t = self.tracks[sid] = AgentTrack(sid, kind)
        return t

    def observe(self, agents_kinds, before, after, moved, kind_of, our_sids=()):
        """One press.  `moved`: sid -> vector this press (things that moved);
        `kind_of`: sprite -> kind.  Things of `agents_kinds` are the agents."""
        b_by = {s.sid: s for s in before}
        a_by = {s.sid: s for s in after}
        for sid, s in a_by.items():
            if sid >= 1000 or sid not in b_by or kind_of(s) not in agents_kinds or sid in our_sids:
                continue
            t = self.track(sid, kind_of(s))
            p, q = b_by[sid], s
            v = tuple(moved.get(sid, (0, 0)))
            v = (int(round(v[0])), int(round(v[1])))
            t.steps.append(v)
            if v != (0, 0):
                m = max(abs(v[0]), abs(v[1]))
                t.stride = m if t.stride is None else min(t.stride, m)
            # THE PREDICTION, checked: the record said where it would stand
            if t.predicted is not None:
                if q.bbox[:2] == t.predicted[:2]:
                    t.hits += 1; t.misses = 0
                elif v != (0, 0):
                    t.misses += 1
                    if t.misses >= 3 and t.want() is not None:
                        # (three broken in a row: the want is re-read from here on —
                        # the record is kept, its weight halved)
                        for kd in t.approach:
                            for k2 in list(t.approach[kd]):
                                t.approach[kd][k2] //= 2
                        t.doubted += 1; t.misses = 0
                t.predicted = None
            # THE WANT: what it closed on this press (by kind, the nearest thing of each kind)
            if v != (0, 0):
                pc, qc = _centre(p.bbox), _centre(q.bbox)
                for kd in {kind_of(o) for o in before if o.sid < 1000 and o.sid != sid and "hud" not in o.tags and "terrain" not in o.tags}:
                    if kd == t.kind:
                        continue
                    near_b = min((_dist(pc, _centre(o.bbox)) for o in before if o.sid < 1000 and o.sid != sid and kind_of(o) == kd and "hud" not in o.tags), default=None)
                    near_a = min((_dist(qc, _centre(o.bbox)) for o in before if o.sid < 1000 and o.sid != sid and kind_of(o) == kd and "hud" not in o.tags), default=None)
                    if near_b is None or near_a is None:
                        continue
                    if near_a < near_b - 0.5:
                        t.approach[kd]["toward"] += 1
                    elif near_a > near_b + 0.5:
                        t.approach[kd]["away"] += 1
            # THE ACT: last press it stood beside a thing of its want; what became of that thing
            if t.beside is not None:
                bsid, bbox0 = t.beside
                if bsid not in a_by:
                    t.acts["took"] += 1
                else:
                    bv = tuple(moved.get(bsid, (0, 0)))
                    if bv != (0, 0):
                        if bv == v and v != (0, 0):
                            t.acts["carried"] += 1
                        else:
                            t.acts["pushed"] += 1
                        # THE SIGN: did the thing go toward the goal's place?
                        if self.goal_place is not None:
                            d0 = _dist(_centre(bbox0), _centre(self.goal_place))
                            d1 = _dist(_centre(a_by[bsid].bbox), _centre(self.goal_place))
                            if d1 < d0 - 0.5:
                                t.sign[bsid] = t.sign.get(bsid, 0) + 1
                            elif d1 > d0 + 0.5:
                                t.sign[bsid] = t.sign.get(bsid, 0) - 1
                    else:
                        t.acts["nothing"] += 1
                t.beside = None
            w = t.want()
            if w is not None:
                near = [o for o in after if o.sid < 1000 and o.sid != sid and kind_of(o) == w and _adjacent(q.bbox, o.bbox)]
                if near:
                    o = min(near, key=lambda o: _dist(_centre(q.bbox), _centre(o.bbox)))
                    t.beside = (o.sid, o.bbox)

    def predict(self, sid, things, kind_of, blocked):
        """Where the agent will stand after its next step: one stride toward
        the nearest thing of its want, the larger axis first, the free way
        (`blocked(cells)` says whether a set of cells may not be entered).
        None when the record says nothing."""
        t = self.tracks.get(sid)
        me = next((s for s in things if s.sid == sid), None)
        if t is None or me is None or t.stride is None:
            return None
        w = t.want()
        if w is None:
            return None
        targets = [o for o in things if o.sid < 1000 and o.sid != sid and kind_of(o) == w and "hud" not in o.tags]
        if not targets:
            return None
        mc = _centre(me.bbox)
        o = min(targets, key=lambda o: _dist(mc, _centre(o.bbox)))
        oc = _centre(o.bbox)
        if _adjacent(me.bbox, o.bbox):
            return me.bbox                 # (beside it: it works there, it does not walk)
        dr, dc = oc[0] - mc[0], oc[1] - mc[1]
        st = t.stride
        steps = sorted([((st if dr > 0 else -st) if abs(dr) >= 0.5 else 0, 0), (0, (st if dc > 0 else -st) if abs(dc) >= 0.5 else 0)],
                       key=lambda v: -(abs(dr) if v[0] else abs(dc)))
        for v in steps:
            if v == (0, 0):
                continue
            cells = {(r + v[0], c + v[1]) for (r, c) in me.cells}
            if not blocked(cells, sid):
                bb = (me.bbox[0] + v[0], me.bbox[1] + v[1], me.bbox[2] + v[0], me.bbox[3] + v[1])
                t.predicted = bb
                return bb
        t.predicted = me.bbox
        return me.bbox

    def describe(self):
        return [t.describe() for t in self.tracks.values()]
