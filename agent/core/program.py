"""A JOURNEY, AND THE PANEL THAT WRITES IT.

Two of the core's principles, composed:

1. WHAT HAPPENS BETWEEN THE PRESS AND THE REST IS PART OF THE ANSWER.  A
   press that returns several frames shows a thing travelling; if by the
   last frame it stands where it stood, the press read by its last frame
   alone was silent — but the thing made a JOURNEY (a sequence of steps)
   and was put back.  Such a press is a LAUNCH.

2. A ROW OF SLOTS IS READ IN ORDER (tr87: a row is a sequence).  When
   the launch's journey has as many steps as the panel has marked slots,
   the slots are the steps, in order: each slot's form is the step it
   makes (a form seen making a step is remembered — a legend learned by
   experiment, not drawn).  A slot's form is changed by clicking the
   marks in it (a click flips a mark).  The journey wanted (the fits
   reading's displacement) is composed of the steps known, and the
   slots are set to write it.

Nothing here names a colour, a size or a place.
"""
from collections import Counter, defaultdict
import numpy as np


def _components(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            cells = []; fr = [(r, c)]; seen[r, c] = True
            while fr:
                y, x = fr.pop(); cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        y2, x2 = y + dy, x + dx
                        if 0 <= y2 < H and 0 <= x2 < W and mask[y2, x2] and not seen[y2, x2]:
                            seen[y2, x2] = True; fr.append((y2, x2))
            out.append(cells)
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def read_journeys(frames, things, hud_rows=()):
    """For each thing (of the frame before the press) that travelled over
    the frames and stands at its start again in the last: its journey,
    the list of step vectors between the distinct places it stood.
    Returns {sid: [steps]}; empty when nothing journeyed and returned."""
    out = {}
    if len(frames) < 3:
        return out
    for s in things:
        if s.size < 3 or "hud" in s.tags or "terrain" in s.tags or s.sid >= 1000:
            continue
        b0 = s.bbox
        if b0[0] in hud_rows or b0[2] in hud_rows:
            continue
        # (the thing's own colour is its body; a part of another colour
        # rides along and is not looked for)
        body_cells = [q for q, col in s.cells.items() if col == s.body] if isinstance(s.cells, dict) else list(s.cells)
        if len(body_cells) < 3:
            continue
        b0 = _bbox(body_cells)
        h, w = b0[2] - b0[0], b0[3] - b0[1]
        form0 = frozenset((r - b0[0], c - b0[1]) for r, c in body_cells)
        places = []
        for f in frames:
            # where does the thing stand in this frame?  On its cells if
            # they wear its colour; else on the like component of its
            # colour and form nearest
            if all(0 <= r < f.shape[0] and 0 <= c < f.shape[1] and int(f[r, c]) == s.body for r, c in body_cells):
                places.append((0, 0)); continue
            best = None
            for comp in _components(f == s.body):
                if len(comp) != len(body_cells):
                    continue
                b = _bbox(comp)
                if (b[2] - b[0], b[3] - b[1]) != (h, w):
                    continue
                if frozenset((r - b[0], c - b[1]) for r, c in comp) != form0:
                    continue
                d = (b[0] - b0[0], b[1] - b0[1])
                if best is None or abs(d[0]) + abs(d[1]) < abs(best[0]) + abs(best[1]):
                    best = d
            if best is None:
                places = None; break
            places.append(best)
        if not places or places[-1] != (0, 0):
            continue
        distinct = [places[0]]
        for p in places[1:]:
            if p != distinct[-1]:
                distinct.append(p)
        if len(distinct) < 3:
            continue                    # (never left, or left and did not return)
        steps = [(distinct[i + 1][0] - distinct[i][0], distinct[i + 1][1] - distinct[i][1]) for i in range(len(distinct) - 2)]
        # (the last leg is the put-back, not a step)
        if steps:
            out[s.sid] = steps
    return out


def read_panel(frame, toggles, bg):
    """THE PANEL: the marks a click flips lie in a strip of one colour;
    along the strip they stand in like slots at one pitch (a slot may be
    empty — the strip's colour throughout).  toggles: [(r, c)] cells of
    marks.  Returns {"box", "axis", "pitch", "slots": [box], "forms":
    [form], "frame_colour"} or None."""
    if len(toggles) < 2:
        return None
    toggles = list(set(toggles))
    H, W = frame.shape
    rs = [r for r, _ in toggles]; cs = [c for _, c in toggles]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    axis = 1 if (c1 - c0) >= (r1 - r0) else 0
    if axis == 0:
        p = read_panel(frame.T, [(c, r) for r, c in toggles], bg)
        if p is None:
            return None
        t = lambda b: (b[1], b[0], b[3], b[2])
        return {"box": t(p["box"]), "axis": 0, "pitch": p["pitch"], "slots": [t(b) for b in p["slots"]],
                "forms": [frozenset((c, r, col) for r, c, col in fm) for fm in p["forms"]], "frame_colour": p["frame_colour"]}
    tset = set(toggles)
    around = Counter()
    for r, c in toggles:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in tset:
                around[int(frame[rr, cc])] += 1
    if not around:
        return None
    fcol = around.most_common(1)[0][0]
    a, c = max(0, r0 - 1), min(H - 1, r1 + 1)
    # the band's busy columns (anything not the strip's colour) as runs
    busy = [x for x in range(W) if any(int(frame[y, x]) != fcol for y in range(a, c + 1))]
    if not busy:
        return None
    runs = []; start = busy[0]; prev = busy[0]
    for x in busy[1:]:
        if x != prev + 1:
            runs.append((start, prev)); start = x
        prev = x
    runs.append((start, prev))
    mark_runs = [rn for rn in runs if any(rn[0] <= cc <= rn[1] for cc in cs)]
    if not mark_runs:
        return None
    width = max(x1 - x0 + 1 for x0, x1 in mark_runs)
    cen = [(x0 + x1) / 2.0 for x0, x1 in mark_runs]
    if len(cen) >= 2:
        gaps = [cen[i + 1] - cen[i] for i in range(len(cen) - 1)]
        pitch = min(gaps)
        if pitch <= 0 or any(abs(g / pitch - round(g / pitch)) > 0.15 for g in gaps):
            return None
        pitch = int(round(pitch))
    else:
        # one mark seen: the pitch is the nearest like-width run's distance, or nothing yet
        others = [rn for rn in runs if rn not in mark_runs and (rn[1] - rn[0] + 1) <= width + 1]
        if not others:
            return None
        pitch = int(round(min(abs((x0 + x1) / 2.0 - cen[0]) for x0, x1 in others)))
        if pitch <= 0:
            return None
    if pitch <= width:
        return None
    half = pitch // 2

    def slot_at(x):
        """The slot centred at x: a mark-like run there, or empty; None
        where the strip ends (a wider thing, or off the screen)."""
        lo, hi = int(round(x)) - half, int(round(x)) - half + pitch - 1
        if lo < 0 or hi >= W:
            return None
        inside = [rn for rn in runs if rn[0] >= lo and rn[1] <= hi]
        crossing = [rn for rn in runs if rn[0] <= hi and rn[1] >= lo and rn not in inside]
        if crossing:
            return None
        if any(rn[1] - rn[0] + 1 > width + 1 for rn in inside):
            return None
        return (a, lo, c, hi)

    slots = {}
    for direction in (-1, 1):
        x = cen[0]
        while True:
            sb = slot_at(x)
            if sb is None:
                break
            slots[int(round(x))] = sb
            x += direction * pitch
    # (an empty slot counts only between marks: the strip's ends are its
    # outermost marks)
    keys_ = sorted(slots)
    def marked(k):
        lo, hi = slots[k][1], slots[k][3]
        return any(rn[0] >= lo and rn[1] <= hi for rn in runs)
    while keys_ and not marked(keys_[0]):
        keys_.pop(0)
    while keys_ and not marked(keys_[-1]):
        keys_.pop()
    if len(keys_) < 2:
        return None
    slot_list = [slots[k] for k in keys_]
    forms = [slot_form(frame, sb, fcol) for sb in slot_list]
    box = (a, slot_list[0][1], c, slot_list[-1][3])
    return {"box": box, "axis": 1, "pitch": pitch, "slots": slot_list, "forms": forms, "frame_colour": fcol}


def read_lattice(frame, toggles, bg, colours=None):
    """THE PANEL AS A LATTICE (2026-10-15, tn36 L2: four columns of slots in
    several rows where L1 had one row): a panel is a region of one colour
    holding like things at one pitch along each axis — a row of slots is the
    lattice with one row.  The columns' pitch is read off the largest group
    of like units (headers, or the marks themselves); the rows' pitch off
    the marks' own groups (the glyphs repeat at the slot's height); slots
    are the tiles, read row-major, each the marks' band tall.  Nothing here
    names a colour or a size: the frame colour is what surrounds the flipped
    marks, the mark colours are what the flips showed, the pitches are what
    the units repeat at."""
    if not toggles:
        return None
    H, W = frame.shape
    tset = set(toggles)
    mark_cols = set(colours or ()) | {int(frame[r, c]) for r, c in toggles}
    around = Counter()
    for r, c in toggles:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in tset:
                around[int(frame[rr, cc])] += 1
    if not around:
        return None
    fcol = around.most_common(1)[0][0]
    mark_cols.discard(fcol)
    seed = next(((r + dr, c + dc) for r, c in toggles for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if 0 <= r + dr < H and 0 <= c + dc < W and int(frame[r + dr, c + dc]) == fcol), None)
    if seed is None:
        return None
    region = set([seed]); fr = [seed]
    while fr:
        y, x = fr.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            y2, x2 = y + dy, x + dx
            if 0 <= y2 < H and 0 <= x2 < W and (y2, x2) not in region and int(frame[y2, x2]) == fcol:
                region.add((y2, x2)); fr.append((y2, x2))
    rs = [r for r, _ in region]; cs = [c for _, c in region]
    a, b, c_, d = min(rs), min(cs), max(rs), max(cs)
    inside = np.zeros((H, W), dtype=bool)
    inside[a:c_ + 1, b:d + 1] = True
    units_mask = inside & (frame != fcol)
    units = []
    for comp in _components(units_mask):
        if all(0 <= r + dr < H and 0 <= c + dc < W and ((r + dr, c + dc) in region or units_mask[r + dr, c + dc])
               for r, c in comp for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            units.append(comp)
    if len(units) < 2:
        return None
    # THE PANEL'S OWN UNITS: those near the flipped mark, and near those, and so
    # on (a gap of a few cells between boxes) — a like-coloured mark across the
    # board is not of this panel
    boxes_all = [_bbox(u) for u in units]
    start_i = next((i for i, u in enumerate(units) if any((r, c) in tset for r, c in u)), None)
    if start_i is None:
        start_i = min(range(len(units)), key=lambda i: min(abs(r - tr_) + abs(c - tc_) for r, c in units[i] for tr_, tc_ in toggles))

    def near(b1, b2, gap=3):
        return not (b1[0] > b2[2] + gap or b2[0] > b1[2] + gap or b1[1] > b2[3] + gap or b2[1] > b1[3] + gap)
    keep = {start_i}; fr = [start_i]
    while fr:
        i = fr.pop()
        for j in range(len(units)):
            if j not in keep and near(boxes_all[i], boxes_all[j]) and len(units[j]) <= 4 * max(len(units[start_i]), 16):
                keep.add(j); fr.append(j)
    units = [units[i] for i in sorted(keep)]
    if len(units) < 2:
        return None
    groups = defaultdict(list)
    for u in units:
        bx = _bbox(u)
        groups[(int(frame[u[0][0], u[0][1]]), bx[2] - bx[0] + 1, bx[3] - bx[1] + 1)].append(bx)

    def pitch_of(vals):
        """The common gap of sorted distinct values (every gap a multiple of the least), else None."""
        vs = sorted(set(vals))
        if len(vs) < 2:
            return None
        gaps = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
        g = min(gaps)
        return g if g > 0 and all(abs(x / g - round(x / g)) < 0.15 for x in gaps) else None
    best_x = None
    for key_, bxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        px_ = pitch_of([bx[1] for bx in bxs])
        if px_ is not None and px_ > max(bx[3] - bx[1] for bx in bxs):
            best_x = (px_, min(bx[1] for bx in bxs), key_[2]); break
    if best_x is None:
        return None
    px, left0, w0 = best_x
    c0 = left0 - (px - w0) // 2          # (a unit sits centred in its slot: the tile starts half the slack before it)
    marks = [bx for key_, bxs in groups.items() if key_[0] in mark_cols for bx in bxs]
    if not marks:
        return None
    py = None
    for key_, bxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if key_[0] not in mark_cols:
            continue
        py_ = pitch_of([bx[0] for bx in bxs])
        if py_ is not None and py_ > max(bx[2] - bx[0] for bx in bxs):
            py = py_; break
    r0 = min(bx[0] for bx in marks); r1 = max(bx[2] for bx in marks)
    if py is None:
        py = r1 - r0 + 1
    n_rows = max(1, int(round((r1 - r0 + 1) / float(py))))
    n_cols = max(1, int((max(bx[3] for bxs in groups.values() for bx in bxs) - c0) // px) + 1)
    slots = []
    for i in range(n_rows):
        for j in range(n_cols):
            top = r0 + i * py; left = c0 + j * px
            slots.append((top, left, min(c_, top + py - 1), min(d, left + px - 1)))
    # (the lattice's extent is its marks': tiles beyond the last marked column or row are not slots)
    last_c = max(bx[3] for bx in marks); last_r = r1
    slots = [sb for sb in slots if sb[1] <= last_c and sb[0] <= last_r]
    if len(slots) < 2:
        return None
    forms = [slot_form(frame, sb, fcol) for sb in slots]
    return {"box": (r0, c0, r1, d), "axis": 1, "pitch": (py, px), "grid": (n_rows, len(slots) // n_rows), "slots": slots,
            "forms": forms, "frame_colour": fcol}


def slot_form(frame, sb, fcol):
    """The form of a slot: its cells not of the frame's colour, by colour,
    relative to the slot's corner."""
    a, bb, c, d = sb
    return frozenset((r - a, cc - bb, int(frame[r, cc])) for r in range(a, c + 1) for cc in range(bb, d + 1) if int(frame[r, cc]) != fcol)


def correspond(forms, journey, known=None, prefix=False):
    """The slots are the steps in order: a reading that assigns to each
    form a step or nothing, such that the marked slots' steps, in order,
    are the journey.  known: {form: step or None} from earlier sightings.
    Returns {form: step or None} or None when no assignment fits.
    prefix: the journey may have ENDED EARLY (a wall, the board's edge — tn36
    L2: twelve like slots, a journey of three) — then the journey is a prefix
    of the slots' steps, and the slots beyond it are left unread."""
    known = dict(known or {})
    n = len(journey)
    if prefix:
        # the first n stepping slots must read as the journey; the rest are unknown
        idx_by_form = defaultdict(list)
        for i, fm in enumerate(forms):
            idx_by_form[fm].append(i)
        assign = dict(known)
        pos = 0
        for i, fm in enumerate(forms):
            if fm in assign and assign[fm] is None:
                continue
            if pos >= n:
                break
            if fm in assign and assign[fm] is not None and tuple(assign[fm]) != tuple(journey[pos]):
                return None
            assign[fm] = tuple(journey[pos]); pos += 1
        return assign if pos == n else None
    # which forms make a step: those not known to be "nothing"; among the
    # unknown, the subsets whose slot count equals the journey length
    idx_by_form = defaultdict(list)
    for i, fm in enumerate(forms):
        idx_by_form[fm].append(i)
    unknown = [fm for fm in idx_by_form if fm not in known]
    fixed_step = [fm for fm in idx_by_form if known.get(fm) is not None]
    fixed_none = [fm for fm in idx_by_form if fm in known and known[fm] is None]
    from itertools import combinations
    for k in range(len(unknown) + 1):
        for sub in combinations(unknown, k):
            stepping = set(fixed_step) | set(sub)
            order = [i for i in range(len(forms)) if forms[i] in stepping]
            if len(order) != n:
                continue
            assign = dict(known)
            ok = True
            for pos, i in enumerate(order):
                fm = forms[i]
                if assign.get(fm) is None and fm in assign:
                    ok = False; break
                if fm in assign and assign[fm] is not None and tuple(assign[fm]) != tuple(journey[pos]):
                    ok = False; break
                assign[fm] = tuple(journey[pos])
            if not ok:
                continue
            for fm in unknown:
                if fm not in stepping:
                    assign[fm] = None
            return assign
    return None


def compose(step_of, want, n_slots, none_form):
    """A journey summing to `want` from at most n_slots steps of the known
    forms (each slot one step, the empty form standing for no step):
    returns [form per slot] or None.  The fewest steps first."""
    steps = {fm: v for fm, v in step_of.items() if v is not None}
    if not steps:
        return None
    from itertools import product
    fms = list(steps)
    best = None
    # bounded search: counts per form
    def rec(i, remaining, acc, total):
        nonlocal best
        if i == len(fms):
            if total == tuple(want) and (best is None or len(acc) < len(best)):
                best = list(acc)
            return
        v = steps[fms[i]]
        for k in range(0, remaining + 1):
            rec(i + 1, remaining - k, acc + [fms[i]] * k, (total[0] + v[0] * k, total[1] + v[1] * k))
    rec(0, n_slots, [], (0, 0))
    if best is None:
        return None
    return best + [none_form] * (n_slots - len(best))


def clicks_to_set(frame, panel, slot_index, form, flips):
    """The clicks that turn the slot's form into `form`: each mark that
    differs is clicked once (a click flips a mark: flips maps a mark's
    colour to what it becomes).  Returns [(r, c)] or None when a differing
    cell is of no mark a click is known to flip."""
    a, bb, c, d = panel["slots"][slot_index]
    fcol = panel["frame_colour"]
    now = slot_form(frame, (a, bb, c, d), fcol)
    now_at = {(r, cc): col for r, cc, col in now}
    want_at = {(r, cc): col for r, cc, col in form}
    diff = [(r, cc) for (r, cc) in set(now_at) | set(want_at) if now_at.get((r, cc), fcol) != want_at.get((r, cc), fcol)]
    if not diff:
        return []
    # differing cells grouped into marks (8-connected)
    diff_set = set(diff); clicks = []
    while diff_set:
        start = diff_set.pop(); comp = {start}; st = [start]
        while st:
            r, cc = st.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, cc + dc)
                    if q in diff_set:
                        diff_set.discard(q); comp.add(q); st.append(q)
        col_now = Counter(now_at.get(q, fcol) for q in comp).most_common(1)[0][0]
        col_want = Counter(want_at.get(q, fcol) for q in comp).most_common(1)[0][0]
        if flips.get(col_now) != col_want and flips.get(col_want) != col_now:
            return None
        rs = sorted(r for r, _ in comp); cs = sorted(cc for _, cc in comp)
        mid = (rs[len(rs) // 2], cs[len(cs) // 2])
        pt = min(comp, key=lambda q: abs(q[0] - mid[0]) + abs(q[1] - mid[1]))
        clicks.append((a + pt[0], bb + pt[1]))
    return clicks
