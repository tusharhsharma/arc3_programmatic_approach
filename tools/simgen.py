"""Concept-composition game generator (simulator v0).

Composes CONCEPTS the agent must hypothesize — control mapping x movement
rule x goal rule x hazard — into small arcengine games, written into
environment_files/ with the `sg` prefix. Purpose (design doc:
claude/concept-simulator-design.md): a DISTRIBUTION of novel tasks to
measure skill-acquisition efficiency (actions-to-first-clear) on held-out
compositions, and a failure categorizer keyed by concept.

Games marked holdout=True use concept COMBINATIONS excluded from the
tuning set — never inspect the agent's behavior on them while developing;
they are the sim's generalization probe.

Usage: python tools/simgen.py [--seed 7] [--out environment_files]
"""
import argparse
import itertools
import json
import random
import sys
from collections import deque
from pathlib import Path

CONTROLS = {
    # ACTION1..4 -> (dx, dy) in cells
    "identity": {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)},
    "rotcw":    {1: (1, 0), 2: (-1, 0), 3: (0, -1), 4: (0, 1)},
    "swapud":   {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)},
}
MOVEMENTS = ("step", "slide", "drift_e")
GOALS = ("reach", "collect3")
HAZARDS = ("none", "static", "timer40")

# v0 sets: tuning combos may be inspected freely; holdout combos are the
# generalization probe (unseen pairings of the SAME concepts).
TUNING = [
    ("identity", "step", "reach", "none"),
    ("identity", "step", "collect3", "static"),
    ("identity", "slide", "reach", "none"),
    ("rotcw", "step", "reach", "none"),
    ("identity", "step", "reach", "timer40"),
    ("identity", "drift_e", "reach", "none"),
    # added after the slide concept landed: the slide x hazard and
    # slide x collect pairings existed ONLY in holdout, so a slide
    # defect there could not be debugged without breaking the holdout
    # discipline — tuning must cover every PAIR we intend to fix
    ("identity", "slide", "reach", "static"),
    ("identity", "slide", "collect3", "none"),
]
HOLDOUT = [
    ("rotcw", "slide", "reach", "static"),
    ("swapud", "step", "collect3", "timer40"),
    ("rotcw", "drift_e", "collect3", "none"),
    ("swapud", "slide", "reach", "timer40"),
    ("identity", "slide", "collect3", "static"),
    ("swapud", "drift_e", "reach", "static"),
]

# SCALE AXIS (the cardinality lesson, measured): the SAME concepts on a
# bigger playground — 20x20 board at 3px cells, more goals, more
# hazards. An agent that solves the small form but not the scaled form
# has memorized the level-1 cardinality, not the concept.
SCALED_TUNING = [
    ("identity", "step", "reach", "none"),
    ("identity", "step", "collect5", "static"),
    ("identity", "slide", "reach", "none"),
]
SCALED_HOLDOUT = [
    ("swapud", "step", "collect5", "static"),
    ("rotcw", "slide", "reach", "none"),
    ("identity", "drift_e", "collect3", "timer40"),
]

W = H = 14          # maze cells (x CELL=4 px => 56x56 in the 64 frame)


def solve_composed(grid, start, goals, hazards, control, movement, timer):
    """BFS under the TRUE composed dynamics — the generator's own first
    lesson: a task must be solvability-PROVEN under its composed rules
    (plain-step reachability certified games that drift made impossible
    and slide made unwinnable). Returns optimal action count or None."""
    from collections import deque as _dq
    Hh, Ww = len(grid), len(grid[0])
    blocked = {(y, x) for y in range(Hh) for x in range(Ww)
               if grid[y][x] == "#"}
    hz = set(hazards)
    gset = frozenset(goals)

    def free(p):
        return (0 <= p[0] < Hh and 0 <= p[1] < Ww and p not in blocked)

    def apply(pos, vec):
        dy, dx = vec[1], vec[0]          # CONTROLS store (dx, dy)
        y, x = pos
        if movement == "slide":
            while free((y + dy, x + dx)):
                y, x = y + dy, x + dx
        else:
            if free((y + dy, x + dx)):
                y, x = y + dy, x + dx
        if movement == "drift_e" and free((y, x + 1)):
            x += 1
        return (y, x)

    vecs = list(CONTROLS[control].values())
    start_state = (start, gset)
    seen = {start_state}
    q = _dq([(start_state, 0)])
    limit = (timer - 3) if timer else 400
    while q:
        (pos, left), d = q.popleft()
        if d >= limit:
            continue
        for v in vecs:
            np_ = apply(pos, v)
            if np_ in hz:
                continue                 # dying is never on the plan
            nleft = left - {np_} if np_ in left else left
            if not nleft:
                return d + 1
            ns = (np_, nleft)
            if ns not in seen:
                seen.add(ns)
                q.append((ns, d + 1))
    return None


def gen_maze(rng, n_goals, n_hazards, control="identity", movement="step",
             timer=0, size=None):
    """Random sparse maze, solvability-proven under the COMPOSED rules."""
    Wl = Hl = size or W
    while True:
        grid = [["." for _ in range(Wl)] for _ in range(Hl)]
        for x in range(Wl):
            grid[0][x] = grid[Hl - 1][x] = "#"
        for y in range(Hl):
            grid[y][0] = grid[y][Wl - 1] = "#"
        for _ in range(rng.randrange(6, 14) * (2 if size else 1)):
            y, x = rng.randrange(2, Hl - 2), rng.randrange(2, Wl - 2)
            L = rng.randrange(2, 5)
            if rng.random() < 0.5:
                for i in range(L):
                    if x + i < Wl - 1:
                        grid[y][x + i] = "#"
            else:
                for i in range(L):
                    if y + i < Hl - 1:
                        grid[y + i][x] = "#"
        free = [(y, x) for y in range(Hl) for x in range(Wl)
                if grid[y][x] == "."]
        rng.shuffle(free)
        if len(free) < 3 + n_goals + n_hazards:
            continue
        py, px = free.pop()
        goals = [free.pop() for _ in range(n_goals)]
        hazards = [free.pop() for _ in range(n_hazards)]
        opt = solve_composed(["".join(r) for r in grid], (py, px),
                             goals, hazards, control, movement, timer)
        if opt is not None:
            grid[py][px] = "P"
            for gy, gx in goals:
                grid[gy][gx] = "G"
            for hy, hx in hazards:
                grid[hy][hx] = "X"
            base = opt + 6      # human baseline ~ optimal + exploration
            return ["".join(row) for row in grid], base


TEMPLATE = '''"""{doc}"""
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, Level, Sprite

CELL = {cellpx}
BG = 0
WALL_C = 9
PLAYER_C = 12
GOAL_C = 8
HAZ_C = 2

CONTROL = {control!r}
MOVEMENT = {movement!r}
GOAL = {goal!r}
HAZARD = {hazard!r}
TIMER = {timer}

MAZES = {mazes}


def _sprites_for(maze):
    sprites = []
    for r, row in enumerate(maze):
        for c, ch in enumerate(row):
            x, y = c * CELL, r * CELL
            if ch == "#":
                sprites.append(Sprite(pixels=[[WALL_C] * CELL for _ in range(CELL)],
                                      name="wall", visible=True, collidable=True,
                                      ).set_position(x, y).set_blocking(BlockingMode.BOUNDING_BOX))
            elif ch == "P":
                sprites.append(Sprite(pixels=[[PLAYER_C] * CELL for _ in range(CELL)],
                                      name="player", visible=True, collidable=True,
                                      ).set_position(x, y).set_blocking(BlockingMode.BOUNDING_BOX))
            elif ch == "G":
                sprites.append(Sprite(pixels=[[GOAL_C] * CELL for _ in range(CELL)],
                                      name="goal", visible=True, collidable=False,
                                      ).set_position(x, y))
            elif ch == "X":
                sprites.append(Sprite(pixels=[[HAZ_C] * CELL for _ in range(CELL)],
                                      name="hazard", visible=True, collidable=False,
                                      ).set_position(x, y))
    return sprites


levels = [Level(sprites=_sprites_for(m), name=f"lv{{i}}")
          for i, m in enumerate(MAZES)]


class {cls}(ARCBaseGame):
    _acts: int = 0

    def __init__(self) -> None:
        super().__init__(game_id={gid!r},
                         levels=levels,
                         camera=Camera(background=BG, letter_box=BG),
                         available_actions=[1, 2, 3, 4])

    def _vec(self):
        m = CONTROL.get(self.action.id.value
                        if hasattr(self.action.id, "value")
                        else int(self.action.id))
        return m

    def step(self) -> None:
        amap = {control_map}
        aid = self.action.id
        key = getattr(aid, "value", None)
        if key is None:
            key = int(aid)
        vec = amap.get(key)
        if vec is not None:
            dx, dy = vec[0] * CELL, vec[1] * CELL
            player = self.current_level.get_sprites_by_name("player")[0]
            if MOVEMENT == "slide":
                for _ in range(16):
                    ox, oy = player.x, player.y
                    self.try_move("player", dx, dy)
                    if (player.x, player.y) == (ox, oy):
                        break
            else:
                self.try_move("player", dx, dy)
            if MOVEMENT == "drift_e":
                self.try_move("player", CELL, 0)
            self._acts += 1
            player = self.current_level.get_sprites_by_name("player")[0]
            for hz in self.current_level.get_sprites_by_name("hazard"):
                if player.x == hz.x and player.y == hz.y:
                    self.lose()
                    self.complete_action()
                    return
            goals = self.current_level.get_sprites_by_name("goal")
            hit = [g for g in goals if player.x == g.x and player.y == g.y]
            if GOAL == "reach":
                if hit:
                    self.next_level()
            else:
                for g in hit:
                    g.visible = False
                    g.collidable = False
                    g.set_position(-100, -100)
                left = [g for g in self.current_level.get_sprites_by_name("goal")
                        if g.x >= 0]
                if not left:
                    self.next_level()
            if TIMER and self._acts >= TIMER:
                self.lose()
        self.complete_action()

    def on_set_level(self, level) -> None:
        self._acts = 0
'''


B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def emit(out_root, idx, combo, holdout, rng, scaled=False):
    # per-game rng keyed by index: adding games never mutates others
    rng = random.Random(rng * 100003 + idx * 7919 + (500000 if scaled else 0))
    control, movement, goal, hazard = combo
    n_goals = {"collect3": 3, "collect5": 5}.get(goal, 1)
    goal_kind = "reach" if goal == "reach" else goal
    n_haz = (6 if scaled else 3) if hazard == "static" else 0
    timer = (60 if scaled else 40) if hazard == "timer40" else 0
    size, cellpx = (20, 3) if scaled else (None, 4)
    mazes, bases = [], []
    for _ in range(2):
        m, b = gen_maze(rng, n_goals, n_haz, control=control,
                        movement=movement, timer=timer, size=size)
        mazes.append(m)
        bases.append(b + (10 if movement != "step" else 0)
                     + (8 if control != "identity" else 0))
    combo_i, inst = divmod(idx, 100)
    tag = "sh" if holdout else "st"
    gid = f"{tag}{B36[combo_i % 36]}{inst}"  # 4 chars: the loader requires it
    cls = gid.capitalize()
    doc = (f"GENERATED concept-composition game: control={control} "
           f"movement={movement} goal={goal} hazard={hazard} "
           f"holdout={holdout} scaled={scaled}")
    src = TEMPLATE.format(doc=doc, control=control, movement=movement,
                          goal=goal_kind, hazard=hazard, timer=timer,
                          mazes=json.dumps(mazes, indent=1),
                          cls=cls, gid=gid, cellpx=cellpx,
                          control_map=repr(CONTROLS[control]))
    d = out_root / f"{gid}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gid}.py").write_text(src)
    meta = {
        "game_id": f"{gid}-sim000000",
        "title": (f"{control}/{movement}/{goal}/{hazard}"
                  + ("/BIG" if scaled else "")),
        "tags": ["sim-composition"] + (["sim-holdout"] if holdout else []),
        "private_tags": [],
        "level_tags": [["sim"], ["sim"]],
        "baseline_actions": bases,
        "date_downloaded": "2026-08-29T00:00:00Z",
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    return gid


INSTANCES = 3   # mazes per combo: single-instance results were maze-luck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="environment_files")
    args = p.parse_args()
    rng = args.seed
    root = Path(args.out)
    made = []
    for i, combo in enumerate(TUNING):
        for k in range(INSTANCES):
            made.append((emit(root, (i + 1) * 100 + k, combo, False, rng),
                         combo, "tuning"))
    for i, combo in enumerate(HOLDOUT):
        for k in range(INSTANCES):
            # idx preserved from the pre-scale generator so the holdout
            # mazes stay IDENTICAL (rng is keyed by idx)
            made.append((emit(root, (i + 50) * 100 + k, combo, True, rng),
                         combo, "HOLDOUT"))
    for i, combo in enumerate(SCALED_TUNING):
        for k in range(INSTANCES):
            made.append((emit(root, (i + 10) * 100 + k, combo, False, rng,
                              scaled=True), combo, "tuning/BIG"))
    for i, combo in enumerate(SCALED_HOLDOUT):
        for k in range(INSTANCES):
            made.append((emit(root, (i + 30) * 100 + k, combo, True, rng,
                              scaled=True), combo, "HOLDOUT/BIG"))
    for gid, combo, kind in made:
        print(f"{gid}  {kind:8} {'/'.join(combo)}")


if __name__ == "__main__":
    main()
