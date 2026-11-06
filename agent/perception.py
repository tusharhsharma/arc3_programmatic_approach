"""Perception v2: translate raw 64x64x16-color frames into structured world state.

Real ARC-AGI-3 games (surveyed 2026-08-17) demand more than naive diffing:
  - HUD elements (energy bars, legends, sequence displays) change on nearly
    every action -> "frame changed" is NOT "the world changed".
  - Some games scroll the camera -> diffs must be computed after estimating
    global translation, and positions kept in world coordinates.
  - Motion can be multi-cell (sliding), objects can morph, appear, vanish.

Pipeline per step:
  raw frame -> playfield detection (static band split)
            -> indicator mask (pixels that change on most actions, learned)
            -> camera-shift estimation
            -> object segmentation + object-level diff (playfield only)
"""
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


# ---------- basic frame handling ----------

def last_grid(obs_frame) -> np.ndarray:
    """The settled state = last animation subframe. Robust to ragged
    subframe lists (np.array of those yields an object array that breaks
    integer indexing downstream)."""
    if isinstance(obs_frame, (list, tuple)):
        if not obs_frame:
            return np.zeros((64, 64), dtype=int)
        return np.asarray(obs_frame[-1], dtype=int)
    arr = np.asarray(obs_frame)
    if arr.ndim == 3:
        return arr[-1].astype(int)
    return arr.astype(int)


def background_color(grid: np.ndarray) -> int:
    return int(Counter(grid.flatten().tolist()).most_common(1)[0][0])


# ---------- lattice detection (kept for tile-based games) ----------

def detect_cell_size(grid: np.ndarray, max_cell: int = 16) -> int:
    h, w = grid.shape
    for p in range(max_cell, 1, -1):
        if h % p or w % p:
            continue
        blocks = grid.reshape(h // p, p, w // p, p)
        if (blocks.min(axis=(1, 3)) == blocks.max(axis=(1, 3))).all():
            return p
    return 1


def downscale(grid: np.ndarray, cell: int) -> np.ndarray:
    """Majority color per cell (mode). Top-left sampling made sub-cell
    detail — like an animated avatar with transparent corners — vanish
    entirely from the logical grid. Ties break to the higher palette index,
    matching the engine's own downscaling convention."""
    if cell == 1:
        return grid
    h, w = grid.shape
    blocks = grid.reshape(h // cell, cell, w // cell, cell) \
                 .transpose(0, 2, 1, 3).reshape(h // cell, w // cell, cell * cell)
    counts = np.stack([(blocks == v).sum(axis=2) for v in range(16)], axis=0)
    return (15 - np.argmax(counts[::-1], axis=0)).astype(grid.dtype)


# ---------- playfield detection ----------

def detect_playfield(grid: np.ndarray) -> tuple:
    """Locate the playfield. Principle: HUD strips / letterbox are uniform
    runs of NON-background color (side bars, full-width progress bars,
    divider lines); uniform runs of the background color are playable empty
    space and belong to the field.

    1. Strip contiguous uniform non-bg rows/cols from each edge.
    2. Split at internal uniform non-bg separator lines; keep the largest
       band if it spans >= 50% of the remainder (avoids fragmenting).
       A DIVIDER that splits a HUD from the field runs edge to edge of the
       FRAME; a line that ends at the room's walls (the strips stripped in
       step 1, of another colour) is a thing inside the room — a wall the
       pieces slide through, a fence — and what lies beyond it is still the
       field (tw03 L2: the pen beyond the line was "HUD" and never read).
    Returns (r0, r1, c0, c1) inclusive-exclusive."""
    bg = background_color(grid)

    def axis_band(g2, axis):
        n = g2.shape[axis]
        if axis == 0:
            lines = [g2[i, :] for i in range(n)]
            full = [grid[i, :] for i in range(n)]
        else:
            lines = [g2[:, i] for i in range(n)]
            full = [grid[:, i] for i in range(n)]
        uniform_nonbg = [bool((ln == ln[0]).all() and ln[0] != bg)
                         for ln in lines]
        uniform_full = [bool((ln == ln[0]).all() and ln[0] != bg)
                        for ln in full]
        lo, hi = 0, n
        while lo < hi - 1 and uniform_nonbg[lo]:
            lo += 1
        while hi > lo + 1 and uniform_nonbg[hi - 1]:
            hi -= 1
        # internal separators: uniform across the whole frame, not only the band
        seps = [i for i in range(lo, hi) if uniform_nonbg[i] and uniform_full[i]]
        if seps:
            cuts = [lo - 1] + seps + [hi]
            best = (lo, hi)
            best_len = -1
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a - 1 > best_len:
                    best_len = b - a - 1
                    best = (a + 1, b)
            if best_len >= (hi - lo) * 0.5:
                lo, hi = best
        return lo, hi

    r0, r1 = axis_band(grid, 0)
    c0, c1 = axis_band(grid[r0:r1], 1)
    return (r0, r1, c0, c1)


# ---------- object segmentation ----------

@dataclass
class Obj:
    id: int
    color: int
    cells: frozenset          # {(row, col), ...}
    bbox: tuple               # (r0, c0, r1, c1) inclusive
    centroid: tuple

    @property
    def size(self):
        return len(self.cells)

    @property
    def shape_key(self):
        r0 = min(r for r, _ in self.cells)
        c0 = min(c for _, c in self.cells)
        return (self.color, frozenset((r - r0, c - c0) for r, c in self.cells))


def segment(logical: np.ndarray, bg: int, mask: np.ndarray = None) -> list:
    """4-connected same-color components, excluding background and masked cells."""
    h, w = logical.shape
    seen = np.zeros((h, w), dtype=bool)
    if mask is not None:
        seen |= mask
    objs = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or logical[r, c] == bg:
                continue
            color = logical[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                            and logical[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [x for x, _ in cells]
            cs = [x for _, x in cells]
            objs.append(Obj(
                id=next_id, color=int(color), cells=frozenset(cells),
                bbox=(min(rs), min(cs), max(rs), max(cs)),
                centroid=(sum(rs) / len(rs), sum(cs) / len(cs)),
            ))
            next_id += 1
    return objs


# ---------- camera shift estimation ----------

def estimate_shift(prev: np.ndarray, cur: np.ndarray, max_shift: int = 16):
    """If a global translation of prev best explains cur, return (dr, dc).
    Returns (0, 0) when the unshifted frames already agree well."""
    if prev.shape != cur.shape:
        return (0, 0)
    h, w = prev.shape
    base = float((prev == cur).mean())
    if base >= 0.90:
        return (0, 0)
    best, best_score = (0, 0), base
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            if dr == 0 and dc == 0:
                continue
            r0p, r1p = max(0, dr), min(h, h + dr)
            c0p, c1p = max(0, dc), min(w, w + dc)
            if r1p - r0p < h // 2 or c1p - c0p < w // 2:
                continue
            a = prev[r0p - dr:r1p - dr, c0p - dc:c1p - dc]
            b = cur[r0p:r1p, c0p:c1p]
            score = float((a == b).mean())
            if score > best_score:
                best, best_score = (dr, dc), score
    if best != (0, 0) and best_score >= max(0.85, base + 0.15):
        return best
    return (0, 0)


# ---------- diffing ----------

@dataclass
class Diff:
    raw_changed: int                      # changed pixels, whole frame
    field_changed: int = 0                # changed cells, playfield minus indicators
    hud_changed: int = 0                  # changes outside playfield / on indicators
    camera_shift: tuple = (0, 0)          # (dr, dc) world translation of the view
    moved: list = field(default_factory=list)      # (color, size, (dr, dc))
    appeared: list = field(default_factory=list)
    vanished: list = field(default_factory=list)
    palette_shift: dict = None                     # canonical->observed map if lighting changed

    @property
    def any_change(self):
        return self.raw_changed > 0

    @property
    def world_changed(self):
        """Did anything meaningful happen in the game world?"""
        return self.field_changed > 0 or self.camera_shift != (0, 0)


def object_diff(prev_objs: list, cur_objs: list) -> tuple:
    """Match objects by shape signature; return (moved, appeared, vanished)."""
    prev_by_shape = {}
    for o in prev_objs:
        prev_by_shape.setdefault(o.shape_key, []).append(o)
    matched_prev = set()
    moved, appeared = [], []
    for o in cur_objs:
        cands = [p for p in prev_by_shape.get(o.shape_key, [])
                 if p.id not in matched_prev]
        if cands:
            p = min(cands, key=lambda p: abs(p.centroid[0] - o.centroid[0])
                    + abs(p.centroid[1] - o.centroid[1]))
            matched_prev.add(p.id)
            dr = round(o.centroid[0] - p.centroid[0])
            dc = round(o.centroid[1] - p.centroid[1])
            if (dr, dc) != (0, 0):
                moved.append((o.color, o.size, (dr, dc)))
        else:
            appeared.append(o)
    cur_shapes = {o.shape_key for o in cur_objs}
    vanished = [p for p in prev_objs
                if p.id not in matched_prev and p.shape_key not in cur_shapes]
    # SHAPE-MORPHING MOVERS: a body can change silhouette WHILE moving (a
    # facing indicator relocating to the leading edge, an animation
    # frame). Exact-shape matching then reports the body as an
    # appear/vanish pair and the only "mover" left is its 1px indicator,
    # whose displacement includes the edge hop — systematically wrong
    # votes on every turn (observed tu93: -8 instead of -6). Pair a
    # vanished object with an appeared one of the SAME COLOR and near
    # size when each is the only instance of that color in its list.
    if appeared and vanished:
        app_by_col = {}
        for o in appeared:
            app_by_col.setdefault(o.color, []).append(o)
        van_by_col = {}
        for p in vanished:
            van_by_col.setdefault(p.color, []).append(p)
        paired_app, paired_van = set(), set()
        for col, apps in app_by_col.items():
            vans = van_by_col.get(col, [])
            if len(apps) == 1 and len(vans) == 1:
                o, p = apps[0], vans[0]
                if 0.75 * p.size <= o.size <= 1.34 * p.size:
                    dr = round(o.centroid[0] - p.centroid[0])
                    dc = round(o.centroid[1] - p.centroid[1])
                    # near pairs only: a collected item vanishing while a
                    # same-color object appears across the map is NOT a
                    # mover (observed: phantom cross-map votes poisoned
                    # the motion table in a collect game)
                    if abs(dr) + abs(dc) > 8:
                        continue
                    if (dr, dc) != (0, 0):
                        moved.append((o.color, o.size, (dr, dc)))
                        paired_app.add(o.id)
                        paired_van.add(p.id)
        if paired_app:
            appeared = [o for o in appeared if o.id not in paired_app]
            vanished = [p for p in vanished if p.id not in paired_van]
    return moved, appeared, vanished


class Perceiver:
    """Stateful perception: locks bg/cell/playfield on the first frame,
    learns indicator pixels online, compensates camera scroll."""

    INDICATOR_MIN_STEPS = 4
    INDICATOR_RATE = 0.6

    def __init__(self):
        self.bg = None
        self.cell = None
        self.playfield = None          # (r0, r1, c0, c1) in logical coords
        self.change_counts = None
        self.steps = 0
        self.cam = (0, 0)              # accumulated camera offset (world = frame + cam)
        self.remap = np.arange(16)     # raw color -> canonical color (palette-shift undo)
        self.palette_shifts = 0

    def reset(self):
        self.__init__()

    # -- frame conversion --

    def logical(self, obs_frame) -> np.ndarray:
        grid = self.remap[last_grid(obs_frame)]
        if self.bg is None:
            self.bg = background_color(grid)
            self.cell = detect_cell_size(grid)
        logical = downscale(grid, self.cell)
        if self.playfield is None:
            self.playfield = detect_playfield(logical)
            self.change_counts = np.zeros(logical.shape, dtype=np.int32)
            self.counts_still = np.zeros(logical.shape, dtype=np.int32)
            self.steps_still = 0
            # per-pixel color memory: how often each pixel displayed each
            # color. A change back to a previously-seen color is routine
            # oscillation (wobbling panels, shimmer); a novel color is a
            # real effect. This lets game content live INSIDE animated
            # regions instead of masking whole areas as HUD.
            self.value_counts = np.zeros((*logical.shape, 16), dtype=np.int16)
            self._count_frame(logical)
        return logical

    def _count_frame(self, logical):
        h, w = logical.shape
        self.value_counts[np.arange(h)[:, None], np.arange(w)[None, :],
                          logical] += 1

    def _routine_mask(self, cur_l):
        """Changed-to-a-color-this-pixel-has-shown-before (>=2 frames)."""
        h, w = cur_l.shape
        counts = self.value_counts[np.arange(h)[:, None],
                                   np.arange(w)[None, :], cur_l]
        return counts >= 2

    def field_view(self, logical: np.ndarray) -> np.ndarray:
        r0, r1, c0, c1 = self.playfield
        return logical[r0:r1, c0:c1]

    @property
    def indicator_mask(self) -> np.ndarray:
        """Pixels behaving like HUD indicators (bars, timers, counters).

        Definition: pixels that change on steps where NO object moved.
        A blocked bump or no-op still ticks the energy bar — that is the
        signature. The avatar can never satisfy it (its pixels change only
        when it moves), so this mask cannot swallow the player, which the
        frequency-based heuristics it replaces repeatedly did."""
        mask = np.zeros_like(self.change_counts, dtype=bool)
        if self.steps_still >= 3:
            mask |= (self.counts_still / self.steps_still) >= 0.5
        return mask

    def _field_mask(self, logical_shape) -> np.ndarray:
        """True where NOT playfield (i.e. excluded)."""
        m = np.ones(logical_shape, dtype=bool)
        r0, r1, c0, c1 = self.playfield
        m[r0:r1, c0:c1] = False
        return m

    # -- objects --

    def objects(self, obs_frame) -> list:
        logical = self.logical(obs_frame)
        return segment(logical, self.bg, mask=self._field_mask(logical.shape))

    # -- diff --

    def diff(self, prev_frame, cur_frame, count_step: bool = True,
             protect=None) -> Diff:
        """protect: logical (r, c) whose 3x3 neighborhood is exempt from
        indicator statistics — an avatar animating in place while blocked
        would otherwise statistically look like a HUD element and be masked
        out of its own world (observed failure mode)."""
        prev_l = self.logical(prev_frame)
        cur_l = self.logical(cur_frame)
        changed_px = prev_l != cur_l
        d = Diff(raw_changed=int(changed_px.sum()))

        # ---- palette-shift detection (lighting / theme change) ----
        # If most pixels changed but a consistent color->color bijection
        # explains them, it's lighting, not physics: fold the mapping into
        # self.remap so all knowledge stays in canonical colors.
        if changed_px.mean() > 0.3:
            mapping = self._try_palette_mapping(prev_l, cur_l, changed_px)
            if mapping:
                inv = {v: k for k, v in mapping.items()}
                new_remap = self.remap.copy()
                for raw in range(16):
                    canon_now = self.remap[raw]
                    if canon_now in inv:
                        new_remap[raw] = inv[canon_now]
                self.remap = new_remap
                self.palette_shifts += 1
                d.palette_shift = mapping
                cur_l = self.logical(cur_frame)     # re-read under new remap
                changed_px = prev_l != cur_l
                d.raw_changed = int(changed_px.sum())

        if count_step:
            self.steps += 1
            self.change_counts[changed_px] += 1

        outside = self._field_mask(prev_l.shape)
        routine = self._routine_mask(cur_l)
        d.hud_changed = int((changed_px & (outside | routine)).sum())

        pf_prev = self.field_view(prev_l)
        pf_cur = self.field_view(cur_l)
        shift = estimate_shift(pf_prev, pf_cur)
        d.camera_shift = shift
        if shift != (0, 0):
            self.cam = (self.cam[0] + shift[0], self.cam[1] + shift[1])
            dr, dc = shift
            h, w = pf_prev.shape
            r0p, r1p = max(0, dr), min(h, h + dr)
            c0p, c1p = max(0, dc), min(w, w + dc)
            aligned_prev = pf_prev[r0p - dr:r1p - dr, c0p - dc:c1p - dc]
            aligned_cur = pf_cur[r0p:r1p, c0p:c1p]
            d.field_changed = int((aligned_prev != aligned_cur).sum())
        else:
            # novel-valued changes inside the playfield = real world events
            mask_pf = (~outside) & (~routine)
            d.field_changed = int((changed_px & mask_pf).sum())

        if d.raw_changed or shift != (0, 0):
            exclude = outside      # segment everything in the playfield —
            prev_objs = segment(prev_l, self.bg, mask=exclude)   # animated
            cur_objs = segment(cur_l, self.bg, mask=exclude)     # regions hold content
            if shift != (0, 0):
                sr, sc = shift
                prev_objs = [Obj(o.id, o.color,
                                 frozenset((r + sr, c + sc) for r, c in o.cells),
                                 (o.bbox[0] + sr, o.bbox[1] + sc,
                                  o.bbox[2] + sr, o.bbox[3] + sc),
                                 (o.centroid[0] + sr, o.centroid[1] + sc))
                             for o in prev_objs]
            d.moved, d.appeared, d.vanished = object_diff(prev_objs, cur_objs)

        # feed the per-pixel color memory with the settled current frame
        if count_step:
            self._count_frame(cur_l)
        return d

    @staticmethod
    def _try_palette_mapping(prev_l, cur_l, changed_px):
        """Return {prev_color: cur_color} if a consistent bijection among the
        DOMINANT changed colors explains >=85% of changed pixels. Small
        residuals (an avatar moving on the same frame as the lighting
        change) are allowed — they remain as genuine world diff after the
        remap is applied."""
        mapping = {}
        explained = 0
        total = int(changed_px.sum())
        if total == 0:
            return None
        for c in np.unique(prev_l[changed_px]):
            sel = changed_px & (prev_l == c)
            n_all = int(sel.sum())
            if n_all < max(8, 0.01 * total):
                continue        # too few pixels to trust (or to matter)
            vals, counts = np.unique(cur_l[sel], return_counts=True)
            top = int(vals[np.argmax(counts)])
            n_top = int(counts.max())
            if n_top < 0.9 * n_all:
                return None     # a dominant color maps inconsistently: not lighting
            mapping[int(c)] = top
            explained += n_top
        if not mapping or explained < 0.85 * total:
            return None
        if len(set(mapping.values())) != len(mapping):
            return None         # colors merging = information loss, reject
        if all(k == v for k, v in mapping.items()):
            return None
        # shift confirmed by dominant colors: extend to small colors too
        # (a goal or avatar re-colored by the same lighting change would
        # otherwise flip-flop each theme and read as a blinking indicator)
        used = set(mapping.values())
        for c in np.unique(prev_l[changed_px]):
            c = int(c)
            if c in mapping:
                continue
            sel = changed_px & (prev_l == c)
            vals = np.unique(cur_l[sel])
            if len(vals) == 1 and int(vals[0]) not in used:
                mapping[c] = int(vals[0])
                used.add(int(vals[0]))
        return mapping

    # -- world coordinates --

    def to_world(self, pos):
        """Frame (logical) coords -> world coords (camera compensated)."""
        return (pos[0] + self.cam[0], pos[1] + self.cam[1])

    def to_frame(self, pos):
        return (pos[0] - self.cam[0], pos[1] - self.cam[1])

    def click_xy(self, logical_pos):
        """Logical playfield coords -> raw pixel x/y for ACTION6."""
        r, c = logical_pos
        return {"x": int(c * self.cell + self.cell // 2),
                "y": int(r * self.cell + self.cell // 2)}
