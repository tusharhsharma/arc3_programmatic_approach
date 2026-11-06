"""LAWS BETWEEN THINGS — the third channel of learning.

The key records say what a press does to a thing.  The frames also say
what holds BETWEEN things, press after press: on every frame of ar25's
first two levels the yellow image was exactly the grey shape mirrored
across the blue axis.  A relation that holds on every frame it can be
tested on is a law; a law lets a thing be DERIVED when it is not on the
screen (the image falls below the frame on the third level) and planned
with as if it were.

Relations tested (a small fixed set, general):
  mirror   b == a reflected across a line thing L (a thin bar spanning
           the frame, vertical or horizontal)
  copy     b == a translated by a fixed vector (a twin)
  turned   b == a turned a quarter or a half about its box (kept for
           later; not derived yet)
Keyed by the KINDS (body colours) of a, b and L: the law is about kinds,
so it carries across levels.  It holds after >= 3 frames with no frame
against it; a frame with a kind-a thing and a kind-L thing but no kind-b
thing that fits counts against it ONLY when a kind-b thing exists at
all (an absent image is what the law is for).
"""
from collections import defaultdict, Counter


def _line_of(s):
    """A thing that is a line across the frame: ('v', column) or ('h', row)
    of its centre, else None."""
    r0, c0, r1, c1 = s.bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    # (a line may be read in pieces where another thing crosses it)
    if w <= 4 and h >= 16 and h >= 4 * w:
        return ("v", (c0 + c1) / 2.0)
    if h <= 4 and w >= 16 and w >= 4 * h:
        return ("h", (r0 + r1) / 2.0)
    return None


def _filled(s):
    """A thing's cells with its enclosed holes filled (an unselected shape
    shows holes where its dots were: the thing is the whole)."""
    cs = set(s.cells)
    try:
        cs |= set(s.cavity())
    except Exception:
        pass
    return cs


def _mirror_cells(cells, line):
    ori, x = line
    out = set()
    for (r, c) in cells:
        if ori == "v":
            out.add((r, int(round(2 * x - c))))
        else:
            out.add((int(round(2 * x - r)), c))
    return out


class ThingLaws:
    def __init__(self):
        self.holds = Counter()          # (rel, kind_a, kind_b, kind_L) -> frames it held
        self.against = Counter()        # frames with the things present and the relation false
        self.copy_vec = {}              # (kind_a, kind_b) -> vector seen

    # ------------------------------------------------------------ observe
    def observe(self, comps):
        things = [s for s in comps if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags
                  and "bar" not in s.tags and "derived" not in s.tags and s.size >= 4]
        lines = [(s, _line_of(s)) for s in comps if s.sid < 1000 and _line_of(s) is not None]
        by_kind = defaultdict(list)
        for s in things:
            by_kind[s.body].append(s)
        for a in things:
            acells = _filled(a)
            # MIRROR across each line
            for L, line in lines:
                if L.sid == a.sid:
                    continue
                img = _mirror_cells(acells, line)
                for kb, bs in by_kind.items():
                    if kb == a.body:
                        continue
                    key = ("mirror", a.body, kb, L.body)
                    if any(_filled(b) == img for b in bs):
                        self.holds[key] += 1
                    elif bs:
                        # (a kind-b thing exists and none is the image: the frame
                        # is against the law — as the docstring says; before, only
                        # a wrong thing AT the image's place counted, so a law
                        # "the pen mirrors the player" held on three coincidences
                        # and was never refuted while the player walked — tw03)
                        self.against[key] += 1
            # COPY: a like-shaped thing of another colour at a fixed offset
            for kb, bs in by_kind.items():
                if kb == a.body:
                    continue
                for b in bs:
                    if b.size == a.size and b.shape == a.shape:
                        v = (b.bbox[0] - a.bbox[0], b.bbox[1] - a.bbox[1])
                        key = ("copy", a.body, kb, None)
                        if self.copy_vec.get((a.body, kb)) in (None, v):
                            self.copy_vec[(a.body, kb)] = v
                            self.holds[key] += 1
                        else:
                            self.against[key] += 1

    # ---------------------------------------------------------------- laws
    def laws(self):
        return [k for k, n in self.holds.items() if n >= 3 and self.against.get(k, 0) == 0]

    def describe(self):
        return [f"{r}: C{b} is C{a}{' across C' + str(L) if L is not None else ''}" for r, a, b, L in self.laws()]

    # -------------------------------------------------------------- derive
    def derive(self, comps, frame_h=64, frame_w=64):
        """Virtual things the laws say exist when no thing of their kind
        does where the law puts it: for a mirror law, the image of every
        kind-a thing across the kind-L thing (cells may lie off the
        frame).  Returned as sprites tagged 'derived', sid 90000 + a.sid."""
        from .entities import Sprite
        out = []
        by_sid = {s.sid: s for s in comps}
        for rel, ka, kb, kL in self.laws():
            if rel != "mirror":
                continue
            lines = [(s, _line_of(s)) for s in comps if s.sid < 1000 and s.body == kL and _line_of(s) is not None]
            if not lines:
                continue
            L, line = lines[0]
            for a in comps:
                if a.sid >= 1000 or a.body != ka or "derived" in a.tags or "hud" in a.tags or "terrain" in a.tags or "bar" in a.tags or a.size < 4:
                    continue
                img = _mirror_cells(_filled(a), line)
                # already on screen as a real thing of kind b?  then nothing to derive
                if any(s.body == kb and s.sid < 1000 and "derived" not in s.tags and len(set(s.cells) & img) >= 0.5 * len(img) for s in comps):
                    continue
                cells = {p: kb for p in img}
                out.append(Sprite(sid=90000 + a.sid, cells=cells, parts=1, tags={"derived"}))
                out[-1].source = (a.sid, L.sid, line)
        return out

    def export(self):
        return {"holds": dict(self.holds), "against": dict(self.against), "copy_vec": dict(self.copy_vec)}

    def seed(self, data):
        if not data:
            return
        for k, n in data.get("holds", {}).items():
            self.holds[k] += n
        for k, n in data.get("against", {}).items():
            self.against[k] += n
        self.copy_vec.update(data.get("copy_vec", {}))
