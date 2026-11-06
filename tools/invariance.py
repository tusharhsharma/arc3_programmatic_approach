"""THE INVARIANCE GAUGE: the same game under a different skin.

A person who has understood a game wins it recoloured, mirrored or
turned on sight; an agent that has understood it does the same, and an
agent fitted to it does not.  `Transformed(env, T)` shows the core a
transformed world and maps its actions back: colours by a bijection,
the frame by a mirror or a quarter turn (keys and click coordinates
mapped along).  Every gap between a game and its transform is a
curve-fit found without a daily slot.

    .venv/bin/python tools/core_play2.py tn36:1 --no-floor --transform colour:7
    .venv/bin/python tools/core_play2.py tn36:1 --no-floor --transform mirror
    .venv/bin/python tools/core_play2.py tn36:1 --no-floor --transform rot90
    .venv/bin/python tools/core_play2.py tn36:1 --no-floor --transform colour:3+mirror

Keys: ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right (the
engine's convention; the core learns each key's effect by observation,
so the mapping only keeps the transformed world faithful to the real
one).  Colour 0 to 15; the bijection never maps a colour to itself
(so a fitted colour is always caught) and is seeded, so a run repeats.
"""
import copy, random
import numpy as np
from arcengine import GameAction

_SWAP_LR = {3: 4, 4: 3}
_SWAP_UD = {1: 2, 2: 1}
# a quarter turn clockwise of the picture: what was "up" on screen is now "right"
_ROT_CW = {1: 4, 4: 2, 2: 3, 3: 1}


def colour_map(seed):
    """A derangement of 0..15 (no colour keeps its number)."""
    rng = random.Random(seed)
    while True:
        p = list(range(16)); rng.shuffle(p)
        if all(p[i] != i for i in range(16)):
            return np.array(p, dtype=np.int64)


class Transform:
    def __init__(self, spec):
        self.spec = spec
        self.cmap = None
        self.mirror = False
        self.rot = 0                     # quarter turns clockwise shown to the core
        for part in spec.split("+"):
            if part.startswith("colour:") or part.startswith("color:"):
                self.cmap = colour_map(int(part.split(":")[1]))
            elif part == "mirror":
                self.mirror = True
            elif part == "flip":
                self.rot = 2
            elif part == "rot90":
                self.rot = 1
            elif part == "rot270":
                self.rot = 3
            elif part:
                raise ValueError(part)

    # ---- the world the core sees
    def frame(self, f):
        f = np.asarray(f)
        if self.cmap is not None:
            f = self.cmap[f]
        if self.mirror:
            f = f[:, ::-1]
        if self.rot:
            f = np.rot90(f, -self.rot)      # (np.rot90 turns counter-clockwise; -k is clockwise)
        return np.ascontiguousarray(f)

    # ---- the core's action, in the real world
    def key(self, aid):
        if self.rot:
            inv = {v: k for k, v in _ROT_CW.items()}
            for _ in range(self.rot):
                aid = inv.get(aid, aid)
        if self.mirror:
            aid = _SWAP_LR.get(aid, aid)
        return aid

    def point(self, x, y, H, W):
        """A click at (x, y) on the transformed picture -> the real picture."""
        # undo the rotation first (the core sees rotation applied last)
        for _ in range(self.rot):
            # the shown picture is the real one turned clockwise; one step back:
            # shown (x, y) on an H x W picture came from real (x', y') = (y, W - 1 - x)
            x, y = y, W - 1 - x
            H, W = W, H
        if self.mirror:
            x = W - 1 - x
        return x, y


class Transformed:
    """An environment wrapper: the core sees the transformed frames; its
    actions are mapped back before the real environment takes them."""

    def __init__(self, env, spec):
        self.env = env
        self.T = Transform(spec)
        self._last = None

    def _wrap(self, obs):
        if obs is None:
            return None
        o = copy.copy(obs)
        frames = [self.T.frame(f) for f in obs.frame]
        o.frame = frames
        self._last = o
        return o

    @property
    def observation_space(self):
        return self._wrap(self.env.observation_space)

    def reset(self):
        return self._wrap(self.env.reset())

    def step(self, action, data=None, **kw):
        aid = action.value
        if aid == 6:
            d = dict(data) if data is not None else {"x": action.action_data.x, "y": action.action_data.y}
            shown = self._last.frame[-1] if self._last is not None else None
            H, W = (shown.shape if shown is not None else (64, 64))
            x, y = self.T.point(int(d["x"]), int(d["y"]), H, W)
            real = GameAction.ACTION6
            real.set_data({"x": int(x), "y": int(y)})
            return self._wrap(self.env.step(real, data={"x": int(x), "y": int(y)}))
        if aid == 0:
            return self._wrap(self.env.step(GameAction.RESET))
        real = GameAction.from_id(self.T.key(aid))
        return self._wrap(self.env.step(real))

    def __getattr__(self, k):
        return getattr(self.env, k)
