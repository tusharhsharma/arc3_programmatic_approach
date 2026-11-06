import numpy as np

from agent.perception import (
    Perceiver, background_color, detect_cell_size, detect_playfield,
    downscale, estimate_shift, segment,
)


def make_grid(cell=4, size=16):
    return np.zeros((size * cell, size * cell), dtype=int)


def paint(g, r, c, color, cell=4):
    g[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = color


def test_background_and_cell_size():
    g = make_grid()
    paint(g, 2, 3, 7)
    paint(g, 5, 5, 9)
    assert background_color(g) == 0
    assert detect_cell_size(g) == 4


def test_cell_size_pixel_game():
    g = np.zeros((64, 64), dtype=int)
    g[10, 13] = 7
    assert detect_cell_size(g) == 1


def test_downscale_and_segment():
    g = make_grid()
    paint(g, 2, 3, 7)
    paint(g, 2, 4, 7)
    paint(g, 8, 8, 9)
    logical = downscale(g, 4)
    objs = segment(logical, bg=0)
    assert len(objs) == 2
    assert next(o for o in objs if o.color == 7).size == 2


def test_playfield_strips_hud_bands():
    g = np.full((64, 64), 5, dtype=int)
    g[60:64, :] = 11          # bottom HUD bar (uniform rows)
    g[0:2, :] = 14            # top bar
    g[20, 20] = 7             # playfield content
    r0, r1, c0, c1 = detect_playfield(g)
    assert r0 >= 2 and r1 <= 60


def test_estimate_shift_detects_scroll():
    rng = np.random.RandomState(0)
    prev = rng.randint(0, 5, (64, 64))
    cur = np.roll(prev, 3, axis=1)      # scroll right by 3
    assert estimate_shift(prev, cur) == (0, 3)


def test_estimate_shift_ignores_small_change():
    prev = np.zeros((64, 64), dtype=int)
    cur = prev.copy()
    cur[10, 10] = 7
    assert estimate_shift(prev, cur) == (0, 0)


def test_diff_motion_and_indicator_learning():
    p = Perceiver()
    base = np.full((64, 64), 0, dtype=int)
    base[30:32, 30:32] = 7            # avatar block
    frames = []
    for i in range(8):
        f = base.copy()
        f[30:32, 30 + i:32 + i] = 0
        f[30:32, 31 + i:33 + i] = 7   # moves right 1px per frame
        f[63, 0:64 - i * 4] = 11      # shrinking "energy bar"
        frames.append(f)
    p.logical(frames[0])
    for i in range(1, 8):
        d = p.diff(frames[i - 1], frames[i])
    # the bar (uniform edge row on frame 0) is outside the playfield;
    # avatar motion is still extracted cleanly
    assert d.moved and d.moved[0][2] == (0, 1)
    assert d.hud_changed > 0


def test_world_changed_false_when_only_hud_ticks():
    p = Perceiver()
    base = np.full((64, 64), 0, dtype=int)
    base[10:12, 10:12] = 7
    frames = []
    for i in range(8):
        f = base.copy()
        f[63, 0:64 - i * 4] = 11      # only the bar changes
        frames.append(f)
    p.logical(frames[0])
    d = None
    for i in range(1, 8):
        d = p.diff(frames[i - 1], frames[i])
    assert not d.world_changed          # bar tick is not a world event
