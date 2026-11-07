"""Gate worlds for the missing-variable pilot.  A player moves on a grid,
may push one box; a pad; a gate cell on the only way to the target.  The
gate's rule depends on history the frame does not show.  Perception is
given (object positions and the gate's displayed state): the pilot tests
rule acquisition and planning only."""
MAP = ["##########",
       "#S...pB..#",
       "#........#",
       "####g#####",
       "#...T....#",
       "##########"]
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
CAP = 15

def parse(m=MAP):
    walls = set(); s = {}
    for r, row in enumerate(m):
        for c, ch in enumerate(row):
            if ch == "#": walls.add((r, c))
            if ch in "SpBgT": s[ch] = (r, c)
    return walls, s["S"], s["p"], s["B"], s["g"], s["T"]

MAP2 = ["###############",
        "#..........pB.#",
        "#S............#",
        "#g#############",
        "#T............#",
        "###############"]
MAP3 = ["#############",
        "#S....p.B...#",
        "#...........#",
        "##########g##",
        "#....T......#",
        "#############"]
MAP4 = ["##############",
        "#p#..........#",
        "#B#..........#",
        "#S...........#",
        "############g#",
        "#T...........#",
        "##############"]
MAP5 = ["###########",
        "#S...#....#",
        "#....g..T.#",
        "#.pB.#....#",
        "#....#....#",
        "###########"]
MAPS = {"small": MAP, "corridor": MAP2, "middle": MAP3, "pocket": MAP4, "rooms": MAP5}
WALLS, START, PAD, BOX, GATE, TARGET = parse()

def use(name):
    global WALLS, START, PAD, BOX, GATE, TARGET
    WALLS, START, PAD, BOX, GATE, TARGET = parse(MAPS[name])

# memory: what history the frame does not show (capped so plans are finite)
# (count of entries capped 3, turns since last entry, turns since last exit, t mod 2, on pad now, last move)
def mem0(onpad, boxpad=False):
    return (1 if onpad else 0, 0 if onpad else CAP, CAP, 0, onpad, None, boxpad, 0)

def mem_step(m, onpad, move, boxpad=False):
    cnt, se, sx, tp, prev, _, _, ex = m
    entered = onpad and not prev; left = prev and not onpad
    cnt = min(3, cnt + (1 if entered else 0))
    se = 0 if entered else min(CAP, se + 1)
    sx = 0 if left else min(CAP, sx + 1)
    ex = min(3, ex + (1 if left else 0))
    return (cnt, se, sx, (tp + 1) % 2, onpad, move, boxpad, ex)

# the true rules (the games)
RULES = {
    "occupancy":   lambda m: m[4],
    "toggle":      lambda m: m[0] % 2 == 1,
    "timer4":      lambda m: m[1] < 4,
    # authored after the learner was frozen:
    "second_entry": lambda m: m[0] >= 2,
    "delay3":      lambda m: m[4] or m[2] < 3,
    "latch":       lambda m: m[0] >= 1,
    "clock":       lambda m: m[3] == 0,
    "last_move_L": lambda m: m[5] == "L",            # outside the learner's primitives
    # corridor versions (timer long enough to be feasible there)
    "timer14":     lambda m: m[1] < 14,
    # NEW held-out, written after learner v2 was frozen:
    "inverse":     lambda m: not m[4],                       # open while the pad is empty
    "brief_second": lambda m: m[0] >= 2 and m[1] < 6,        # second entry opens it, briefly
    "box_only":    lambda m: m[6],                           # only the box on the pad counts (no primitive)
    "exits_odd":   lambda m: m[7] % 2 == 1,                  # each exit toggles (no exact primitive)
    "timer6":      lambda m: m[1] < 6,
    "timer13":     lambda m: m[1] < 13,
}
DEV = ["occupancy", "toggle", "timer4"]
HELD = ["second_entry", "delay3", "latch", "clock", "last_move_L"]

def step(pos, box, m, gate_open, a):
    """One action under the world's physics (known prior: walls stop,
    contact pushes the box, a closed gate stops)."""
    dr, dc = MOVES[a]; n = (pos[0] + dr, pos[1] + dc)
    if n in WALLS or (n == GATE and not gate_open):
        n = pos
    elif n == box:
        b2 = (box[0] + dr, box[1] + dc)
        if b2 in WALLS or b2 == GATE: n = pos
        else: box = b2
    onpad = PAD in (n, box)
    return n, box, mem_step(m, onpad, a, box == PAD)

class Game:
    def __init__(self, rule):
        self.rule = RULES[rule]; self.pos, self.box = START, BOX
        self.m = mem0(PAD in (START, BOX), BOX == PAD); self.actions = 0
    def gate(self): return bool(self.rule(self.m))
    def obs(self): return {"pos": self.pos, "box": self.box, "gate": self.gate()}
    def act(self, a):
        self.pos, self.box, self.m = step(self.pos, self.box, self.m, self.gate(), a)
        self.actions += 1
        return self.obs()
    def won(self): return self.pos == TARGET

def oracle_len(rule, limit=200000):
    """Shortest solution knowing the rule and carrying full memory."""
    from collections import deque
    f = RULES[rule]; s0 = (START, BOX, mem0(PAD in (START, BOX), BOX == PAD))
    q = deque([(s0, 0)]); seen = {s0}
    while q:
        (p, b, m), d = q.popleft()
        if p == TARGET: return d
        for a in "UDLR":
            s = step(p, b, m, bool(f(m)), a)
            if s not in seen and len(seen) < limit:
                seen.add(s); q.append((s, d + 1))
    return None
