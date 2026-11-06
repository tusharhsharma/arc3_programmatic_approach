"""THE RUN'S TELLING IS A GOAL AND A TEST (statement 6 of the no-reading
class, 2026-10-27; read from the human replay of sp80).

A key whose press draws a TRACE over the board — ink of one colour that
starts beside a thing (the source), runs on from frame to frame, paints
things on its way and enters the hollows of others — and then takes it
all back, has told three things:

  * the things that FLASH at the end are what the run is about: the
    kind to fill (sp80's cups blink when the water missed them);
  * the things the trace ENTERED are the ones this run filled — the
    count of them, among the kind to fill, is the run's score;
  * the things the trace PAINTED on its way (recoloured whole) are what
    bends it — the mover among them is what the plan may place.

And the trace itself is a LAW the plan can run on its own: the ink runs
straight from its source; what it meets, it runs along to the ends of
and falls off both; what it enters, it fills.  The law is trusted only
when it redraws every trace seen so far; when it does not, the plan
falls back to the test — one run per place, nearest first.

Nothing here names a game, a colour, or a coordinate.
"""
from collections import Counter

import numpy as np

from agent.core.progress import _components


def _rays_inside(cells, bbox):
    """Free cells of a thing's box that its own cells shut in on at least
    three of the four sides — the inside of a cup, a pocket, a tube open
    at one end (the box-cavity reading needs all four)."""
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return frozenset()
    rows = {}
    cols = {}
    for (r, c) in cells:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)
    out = set()
    for r in range(r0, r1 + 1):
        rc = rows.get(r, [])
        for c in range(c0, c1 + 1):
            if (r, c) in cells:
                continue
            cr = cols.get(c, [])
            n = (any(x < c for x in rc) + any(x > c for x in rc)
                 + any(x < r for x in cr) + any(x > r for x in cr))
            if n >= 3:
                out.add((r, c))
    return frozenset(out)


def read_run(frames, before, f_before, hud_rows=(), bg=None):
    """What a multi-frame press drew and took back.  Returns None when no
    trace was drawn; else a dict:
      colour     the ink
      trace      cells in order of first appearance (not a thing's own)
      source     sid of the thing of the ink's colour the trace starts
                 beside (None when none)
      direction  unit (dr, dc) of the trace's first step from the source
      painted    sids recoloured whole on the way, in order
      entered    sids whose inside the trace reached, in order
      flash      sids flipping over the tail
      restored   the last frame is (nearly) the first
      n          frames
    """
    n = len(frames)
    if n < 4:
        return None
    H, W = f_before.shape
    hud = set(hud_rows)
    own = {}
    for s in before:
        if "hud" in s.tags or s.sid >= 1000:
            continue
        for p in s.cells:
            own[p] = s.sid
    first = {}
    for i in range(n - 1):              # (the last frame is the undoing)
        d = np.argwhere(frames[i] != f_before)
        if len(d) > 1500:
            continue
        for r, c in d:
            p = (int(r), int(c))
            if p[0] in hud or p in own or p in first:
                continue
            first[p] = (i, int(frames[i][p]))
    if len(first) < 4:
        return None
    colour = Counter(v[1] for v in first.values()).most_common(1)[0][0]
    trace = sorted((p for p, v in first.items() if v[1] == colour), key=lambda p: (first[p][0], p))
    if len(trace) < 4:
        return None
    # the source: a thing of the ink's colour beside the first cells
    head = [p for p in trace if first[p][0] == first[trace[0]][0]]
    # the sources: every thing of the ink's colour beside the first cells
    # (three taps pour three streams)
    sources = []
    for p in head:
        for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
            sid = own.get(q)
            if sid is not None and sid not in sources:
                s = next((x for x in before if x.sid == sid), None)
                if s is not None and s.body == colour:
                    sources.append(sid)
    source = sources[0] if sources else None
    direction = None
    if source is not None:
        s = next(x for x in before if x.sid == source)
        sc = s.centre
        near = [p for p in head if abs(p[0] - sc[0]) + abs(p[1] - sc[1]) <= max(s.dims) + 2] or head
        hc = (sum(p[0] for p in near) / len(near), sum(p[1] for p in near) / len(near))
        dr, dc = hc[0] - sc[0], hc[1] - sc[1]
        direction = (int(np.sign(dr)), 0) if abs(dr) >= abs(dc) else (0, int(np.sign(dc)))
    # painted: things recoloured whole at some frame
    painted = []
    for s in before:
        if "hud" in s.tags or s.sid >= 1000 or s.size < 2:
            continue
        cells = list(s.cells.keys())
        for i in range(n - 1):
            ch = sum(1 for p in cells if frames[i][p] != f_before[p])
            if ch * 2 >= len(cells):
                painted.append((i, s.sid))
                break
    painted = [sid for _, sid in sorted(painted)]
    # entered: things whose inside the trace reached
    tset = set(trace)
    entered = []
    for s in before:
        if "hud" in s.tags or s.sid >= 1000 or s.size < 4:
            continue
        ins = _rays_inside(set(s.cells.keys()), s.bbox)
        hit = ins & tset
        if hit:
            entered.append((min(first[p][0] for p in hit), s.sid))
    entered = [sid for _, sid in sorted(entered)]
    # flash: things flipping over the tail
    tail = min(8, n)
    flips = Counter()
    for i in range(n - tail + 1, n):
        d = np.argwhere(frames[i] != frames[i - 1])
        for r, c in d:
            p = (int(r), int(c))
            if p[0] not in hud:
                flips[p] += 1
    flash = []
    for s in before:
        if "hud" in s.tags or s.sid >= 1000 or s.size < 4:
            continue
        k = sum(1 for p in s.cells if flips.get(p, 0) >= 2)
        if k * 2 >= s.size:
            flash.append(s.sid)
    # taken back: the trace's cells wear what they wore (other things may
    # have changed — a choice reset by the run — and still the ink is gone)
    d_end = {(int(r), int(c)) for r, c in np.argwhere(frames[-1] != f_before) if int(r) not in hud}
    restored = len(d_end & set(trace)) <= max(2, len(trace) // 10)
    return {"colour": colour, "trace": trace, "source": source, "sources": sources, "direction": direction,
            "painted": painted, "entered": entered, "flash": flash, "restored": restored, "n": n}


def flow(frame, source_cells, direction, bg, ink, hud_rows=(), max_steps=4000, sink=()):
    """THE TRACE AS A LAW.  The ink leaves the source along `direction`
    with the source's breadth and runs straight over the background.  At
    anything else it spreads along the near face, both ways, as far as
    the face has the thing beneath it; where the face is open beyond, it
    falls there with the same breadth; where the face itself is shut, it
    stops.  At a SINK (the board's own ground — what spans its edge) the
    ink is lost: it stops there, touching it.  Returns (inked cells,
    touched cells of the things it met)."""
    H, W = frame.shape
    hud = set(hud_rows)
    sink = set(sink)
    dr, dc = direction
    src = set(source_cells)
    if dr:
        A = lambda p: p[0] * dr          # along-position (grows with the flow)
        L = lambda p: p[1]               # lateral position
        P = lambda a, l: (a * dr, l)     # back to a cell (dr is +-1)
        nA, nL = H, W
    else:
        A = lambda p: p[1] * dc
        L = lambda p: p[0]
        P = lambda a, l: (l, a * dc)
        nA, nL = W, H

    def ok(p):
        return 0 <= p[0] < H and 0 <= p[1] < W and p[0] not in hud

    def open_(p):
        return ok(p) and int(frame[p]) in (bg, ink)

    a_edge = max(A(p) for p in src)
    lats = sorted(L(p) for p in src if A(p) == a_edge)
    breadth = max(1, len(lats))
    inked = set(); touched = set()
    queue = [(a_edge + 1, lats[0], lats[-1])]
    seen = set()
    steps = 0
    while queue and steps < max_steps:
        a, l0, l1 = queue.pop(0)
        if (a, l0, l1) in seen:
            continue
        seen.add((a, l0, l1))
        while steps < max_steps:
            steps += 1
            cells = [P(a, l) for l in range(l0, l1 + 1)]
            if not all(ok(p) for p in cells):
                break
            blocked = [p for p in cells if not open_(p)]
            if not blocked:
                inked.update(cells)
                a += 1
                continue
            touched.update(blocked)
            if any(p in sink for p in blocked):
                break                               # lost in the ground
            face = a - 1
            # the band along the face, both ways
            for side in (-1, 1):
                l = (l0 - 1) if side < 0 else (l1 + 1)
                while 0 <= l < nL:
                    pf = P(face, l)
                    if not open_(pf):
                        break                       # the face is shut: a wall across
                    pb = P(face + 1, l)
                    if ok(pb) and not open_(pb):
                        for k in range(breadth):    # the band is the stream's breadth thick
                            q = P(face - k, l)
                            if open_(q):
                                inked.add(q)
                        touched.add(pb)
                        l += side
                        continue
                    # open beyond: the ink falls here with its breadth
                    if side < 0:
                        queue.append((face - breadth + 1, max(0, l - breadth + 1), l))
                    else:
                        queue.append((face - breadth + 1, l, min(nL - 1, l + breadth - 1)))
                    break
            break
    return inked, touched


def entered_by(inked, things):
    """The things whose inside the ink reached."""
    out = []
    for s in things:
        if "hud" in s.tags or s.sid >= 1000 or s.size < 4:
            continue
        ins = _rays_inside(set(s.cells.keys()), s.bbox)
        if ins & inked:
            out.append(s.sid)
    return out


def sink_of(things):
    """The board's own ground: what is read as terrain."""
    out = set()
    for s in things:
        if "terrain" in s.tags:
            out.update(s.cells.keys())
    return out


def flow_all(frame, run, things, bg, hud_rows=()):
    """The law run from every source of the run (the sources' cells as
    they stand in `things`)."""
    inked = set(); touched = set()
    sink = sink_of(things)
    for sid in run.get("sources") or [run["source"]]:
        src = next((s for s in things if s.sid == sid), None)
        if src is None:
            continue
        a, b = flow(frame, src.cells.keys(), run["direction"], bg, run["colour"], hud_rows, sink=sink)
        inked |= a; touched |= b
    return inked, touched


def law_holds(frame, run, things, bg, hud_rows=()):
    """Does the law redraw this run's trace?  (Jaccard of the inked cells
    against the trace seen; the ink over the things' own cells is not
    counted on either side.)"""
    if run["source"] is None or run["direction"] is None:
        return False
    inked, _ = flow_all(frame, run, things, bg, hud_rows)
    seen = set(run["trace"])
    if not inked:
        return False
    j = len(inked & seen) / float(len(inked | seen))
    return j >= 0.8
