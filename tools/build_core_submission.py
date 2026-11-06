"""Bundle THE CORE (the fiddling scientist on the scene tracker) into one
my_agent.py for the official ARC-AGI-3 starter kit.

Unlike tools/build_submission.py (which concatenated the old agent's
modules into one namespace and had to police name collisions), this
bundler EMBEDS each module's source and loads it at import time into a
proper module object registered under its real name (agent.core.planner,
tools.core_play2, ...), so package-relative and function-local imports
work untouched and nothing collides.

The adapter (MyAgent) inverts control: the core drives an environment
(`env.step(action)`, `env.reset()`) from a worker thread; the harness
calls `choose_action(frames, latest_frame)`.  A bridge hands the core
each frame as the answer to its last action and returns the core's next
action to the harness.

    .venv/bin/python tools/build_core_submission.py
    .venv/bin/python tools/build_core_submission.py --check   # plays a level through the adapter offline
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "agent/__init__.py",
    "agent/perception.py",
    "agent/goals.py",
    # THE FLOOR: the crude loop (v11's agent), the standing policy under the core
    "agent/agents.py",
    "agent/tracker.py",
    "agent/world_model.py",
    "agent/transform_solver.py",
    "agent/docking.py",
    "agent/multibody.py",
    "agent/permuter.py",
    "agent/pegs.py",
    "agent/aligner.py",
    "agent/scene_analysis.py",
    "agent/mapper_agent.py",
    "agent/core/__init__.py",
    "agent/core/claims.py",
    "agent/core/journal.py",
    "agent/core/progress.py",
    "agent/core/agents.py",
    "agent/core/runkey.py",
    "agent/core/lever.py",
    "agent/reader.py",
    "agent/core/perthread.py",
    "agent/core/ledger.py",
    "agent/core/planner.py",
    "agent/core/entities.py",
    "agent/core/scene.py",
    "agent/core/thinglaws.py",
    "agent/core/fiddle.py",
    "agent/core/relations.py",
    "agent/core/program.py",
    "agent/core/leap.py",
    "agent/core/table.py",
    "agent/core/legend.py",
    "agent/core/rules.py",
    "tools/__init__.py",
    "tools/core_play2.py",
]

HEADER = '''"""ARC Prize 2026 — ARC-AGI-3 submission: THE CORE OVER THE FLOOR.

A fully symbolic agent, no neural networks and no game-specific code.
THE CORE: a scene tracker (opaque objects that persist, frames that
render them), a fiddling scientist (every key, a click on each distinct
thing, the silent keys in contact) whose observations become laws
stated over kinds and forms (translate / extend / push / select /
teleport / carry / leap / launch / paint), first-frame readings from a
relation language (fits, covers, like marks onto like marks, a picture
beside a canvas, a table of pairs), and a planner that composes the
induced laws and verifies each step by use.  THE FLOOR under it: the
crude loop (probe, the unique thing, the frontier, class-scored clicks,
an escalating death ledger, random known motion) that plays every level
the core leaves, from where it stands, and never gives a level up while
the game runs.  Auto-generated bundle — edit the modules in the arc3
repo and re-run tools/build_core_submission.py.
"""
import sys, types, threading, queue, time, random, zlib, base64, json

# (the modules' sources, compressed: the bundle stays one file well under
# Kaggle's kernel size limit)
_SOURCES_Z = %(sources)s


def _load_modules():
    for path, src in json.loads(zlib.decompress(base64.b64decode(_SOURCES_Z)).decode("utf-8")):
        name = path[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[:-9]
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__file__ = path
        if path.endswith("__init__.py"):
            mod.__path__ = []
            mod.__package__ = name
        else:
            mod.__package__ = name.rpartition(".")[0]
        sys.modules[name] = mod
        parent = name.rpartition(".")[0]
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], name.rpartition(".")[2], mod)
        exec(compile(src, path, "exec"), mod.__dict__)


_load_modules()

from arcengine import GameAction, GameState
from tools.core_play2 import play_game

# THE SEAT CLOCK: time, as the core sees it, ticks only while its thread
# holds the seat — the core's clocks (a level's turn, a plan search, the
# game's budget) are then thinking time, not the hours a game spends
# waiting its turn between actions
_time = time
_WAITED = {}                           # thread ident -> seconds spent not holding the seat


class _SeatClock:
    def __getattr__(self, k):
        return getattr(_time, k)

    def time(self):
        return _time.time() - _WAITED.get(threading.get_ident(), 0.0)


for _m in ("tools.core_play2", "agent.core.planner"):
    sys.modules[_m].time = _SeatClock()
'''

ADAPTER = '''

class _Stop(BaseException):
    """Raised inside the worker when the harness has stopped asking: it
    passes every `except Exception` in the core and frees the core lock."""


class _FairSeat:
    """One seat, taken in the order asked (a queue, not a scramble), and
    abandoned to the next in line when a holder keeps it past HOLD_MAX
    (a hung thread must not stop every other game)."""
    HOLD_MAX = 45.0                     # (hand-chosen: one seat is the whole run's flow of actions, and the gateway must never see minutes of silence — measured on the public games, single thinks of 90–170 s occur; past this the next in line thinks alongside, and the long thinker finishes on its own time)

    def __init__(self):
        self.cv = threading.Condition()
        self.q = []
        self.holder = None
        self.since = 0.0

    def acquire(self, deadline=None):
        me = object()
        with self.cv:
            self.q.append(me)
            while True:
                free = self.holder is None or (_time.time() - self.since > self.HOLD_MAX)
                if free and self.holder is not None and self.q[0] is me:
                    sys.stderr.write(f"[seat] abandoned after {_time.time() - self.since:.0f}s\\n")
                if free and self.q[0] is me:
                    self.q.pop(0)
                    self.holder = me
                    self.since = _time.time()
                    return me
                if deadline is not None and _time.time() > deadline:
                    self.q.remove(me)
                    return None
                self.cv.wait(2.0)

    def release(self, token):
        with self.cv:
            if self.holder is token:
                self.holder = None
            self.cv.notify_all()


_SEAT = _FairSeat()


class _Bridge:
    """The environment the core drives, from the harness's side: an
    action goes out, the frame that answers it comes back.  The seat is
    given up while the frame is awaited and taken again (in turn) before
    the core thinks on; the wait is not the core's time."""

    def __init__(self):
        self.acts = queue.Queue()
        self.obs = queue.Queue()
        self.last = None                # the frame that answered the last action
        self.closed = False             # the harness will not ask again
        self.token = None               # the seat, while held
        self.held_since = None          # wall time the seat was last taken (for the watchdog)

    def take_seat(self):
        t0 = _time.time()
        self.token = _SEAT.acquire(deadline=_DEADLINE)
        _WAITED[threading.get_ident()] = _WAITED.get(threading.get_ident(), 0.0) + (_time.time() - t0)
        self.held_since = _time.time() if self.token is not None else None
        return self.token is not None

    def leave_seat(self):
        if self.token is not None:
            _SEAT.release(self.token)
            self.token = None
            self.held_since = None

    def step(self, action, data=None):
        if self.closed:
            raise _Stop()
        if data is not None and hasattr(action, "set_data"):
            action.set_data(data)
        self.leave_seat()
        self.acts.put(action)
        o = self.obs.get()
        if not self.take_seat():
            raise _Stop()
        if o is None or self.closed:
            raise _Stop()
        self.last = o
        return o

    def reset(self):
        return self.step(GameAction.RESET)


try:
    from agents.agent import Agent
except Exception:                       # (offline check without the harness)
    class Agent:
        def __init__(self, *a, **k):
            pass


# THE SEAT: EVERY GAME AT ONCE, ONE THINKING AT A TIME.  The harness starts
# every game in its own thread (~110 in the rerun) and the core's thinking
# is Python under one interpreter lock, so one seat is the CPU: a game
# takes it to think out its next action, gives it up while the gateway
# answers, and takes it again in turn — every game acts from the first
# minute and keeps acting to the end (a game left idle for hours, as the
# whole-game seats of v35 did to a hundred games, scored 0.00 on Kaggle;
# the floor-first seating of v34 fed the core nothing).  A game's budget
# is seat time — the CPU's share — not wall time.
import os as _os
_SEATS = 1
_GAMES = {"made": 0, "done": 0}
_GAMES_LOCK = threading.Lock()
# THE CLOCK: the run's budget is one number, kept well under the notebook's
# limit; a game's share of seat time is the seat's time divided among the
# games (Kaggle: 27 000 s / 110 ≈ 245 s of thinking a game)
_RUN_START = time.time()
_RUN_BUDGET_S = float(_os.environ.get("ARC_RUN_BUDGET_S", str(int(7.5 * 3600))))    # (the notebook's limit is 9 h = 32400 s; 7.5 h leaves the rerun's setup and teardown their margin.  A 2 h run of this seating failed at 42 min as the 7.5 h runs did: the fault does not grow with the run)
# RESET keeps the level: the same flag the Duck harness pins before the client is built
_os.environ.setdefault("ONLY_RESET_LEVELS", "true")
_DEADLINE = _RUN_START + _RUN_BUDGET_S


def _all_done():
    with _GAMES_LOCK:
        return _GAMES["done"] >= _GAMES["made"]


def _slice():
    """This game's share of seat time: the seat's time left divided among
    the games not yet done (this one included)."""
    with _GAMES_LOCK:
        waiting = max(1, _GAMES["made"] - _GAMES["done"])
    return max(60.0, (_DEADLINE - _time.time() - 60.0) * _SEATS / waiting)


class MyAgent(Agent):
    """The core behind the ARC-AGI-3-Agents contract."""

    MAX_ACTIONS = 4000
    ACTION_WAIT_S = 240                 # the longest a worker may keep the seat without an action (a plan search is bounded at 90 s; past HOLD_MAX the seat has moved on, and past this the game plays on without its stuck thinker — and keeps the gateway's clock alive)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bridge = None
        self._thread = None
        self._t0 = None
        self._pending = False           # an action is out; the next frame answers it
        self._dead = False
        self._n = 0                     # actions handed to the harness
        self._seated = False            # the worker holds a seat (the game is being played)
        self._over = False              # the worker has finished: nothing more to say
        with _GAMES_LOCK:
            _GAMES["made"] += 1

    def is_done(self, frames, latest_frame):
        done = latest_frame.state is GameState.WIN or time.time() > _DEADLINE or (self._over and _all_done())
        if done:
            self._stop()
        return done

    def _stop(self):
        """The harness will not ask again: wake the worker so it leaves
        the core (and frees its seat for the next game)."""
        if self._bridge is not None:
            self._bridge.closed = True
            if self._pending:
                self._pending = False
                self._bridge.obs.put(None)

    def append_frame(self, frame):
        # (the harness keeps every frame of every game; only the last is
        # read — ~110 games of frames would be gigabytes by the end)
        super().append_frame(frame)
        if len(self.frames) > 4:
            del self.frames[:-4]

    def main(self):
        # (a gateway stall ends the harness's loop with an exception: the
        # seat must not go with it)
        try:
            super().main()
        finally:
            self._over = True
            self._stop()

    def do_action_request(self, action):
        """The harness's request with one difference: a frame that does
        not come (a request past the gateway's 10 s timeout returns None)
        is asked for again before the game is given up."""
        data = action.action_data.model_dump()
        raw = None
        for i in range(3):
            raw = self.arc_env.step(action, data=data, reasoning=None)
            if raw is not None:
                break
            time.sleep(2.0 + 3.0 * i)
        return self._convert_raw_frame_data(raw)

    def _start(self, obs):
        self._bridge = _Bridge()
        b = self._bridge

        def run():
            try:
                _WAITED[threading.get_ident()] = 0.0
                if not b.take_seat():
                    return
                self._seated = True
                self._t0 = _time.time()
                play_game(b, obs, 40, False, 1, None, (), getattr(obs, "game_id", "game")[:4], persist=True,
                          floor=True, max_actions=self.MAX_ACTIONS, seconds=min(_slice(), max(60.0, _DEADLINE - _time.time() - 60.0)), lock=None)
            except _Stop:
                pass
            except Exception as e:
                sys.stderr.write(f"[core] worker ended: {e!r}\\n")
            finally:
                b.leave_seat()
                with _GAMES_LOCK:
                    _GAMES["done"] += 1
                b.acts.put(None)          # (the core has nothing more to say)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self._pending = False

    def _random(self, latest_frame):
        avail = [a for a in getattr(latest_frame, "available_actions", []) or [] if a not in (0, 7)]
        if not avail:
            avail = [1, 2, 3, 4, 5, 6]
        aid = random.choice(avail)
        act = GameAction.from_id(aid)
        if act.is_complex():
            act.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        return act

    def _next_action(self):
        while True:
            try:
                return self._bridge.acts.get(timeout=30)
            except queue.Empty:
                # (a game waits its turn for as long as it takes; only a
                # worker that keeps the seat and says nothing is a fault)
                since = self._bridge.held_since
                if since is not None and _time.time() - since > self.ACTION_WAIT_S:
                    raise RuntimeError("the core is silent")

    def choose_action(self, frames, latest_frame):
        if latest_frame.state is GameState.NOT_PLAYED:
            return GameAction.RESET
        if self._dead:
            if latest_frame.state is GameState.GAME_OVER:
                return GameAction.RESET
            return self._random(latest_frame)
        if self._over:
            # THE GAME'S TURN IS OVER: the seat went to the next game; this
            # thread waits out the run without acting (a random action now
            # would only load the gateway the seated games are talking to),
            # and the one action it must give at the deadline ends the loop
            while time.time() < _DEADLINE and not _all_done():
                time.sleep(10)
            time.sleep(random.uniform(0.0, 90.0))    # (the games end together: their last RESETs are spread, not a burst at the gateway)
            return GameAction.RESET
        try:
            if self._bridge is None:
                self._start(latest_frame)
            elif self._pending:
                self._bridge.obs.put(latest_frame)
            act = self._next_action()
            if act is None:
                self._over = True
                self._pending = False
                return self.choose_action(frames, latest_frame)
            self._pending = True
            self._n += 1
            if self._n >= self.MAX_ACTIONS:
                self._stop()            # (the harness stops at MAX_ACTIONS without asking is_done)
            return act
        except Exception as e:
            sys.stderr.write(f"[core] adapter fallback: {e!r}\\n")
            self._dead = True
            self._stop()                # (the worker, when it next steps, leaves; its seat is not kept)
            if latest_frame.state is GameState.GAME_OVER:
                return GameAction.RESET
            return self._random(latest_frame)
'''


def build():
    sources = []
    for m in MODULES:
        p = ROOT / m
        src = p.read_text() if p.exists() else ""
        sources.append((m, src))
    import zlib, base64
    blob = base64.b64encode(zlib.compress(json.dumps(sources).encode("utf-8"), 9)).decode("ascii")
    lines = [blob[i:i + 100] for i in range(0, len(blob), 100)]
    bundle = HEADER % {"sources": "(\n" + "\n".join(f'    "{l}"' for l in lines) + "\n)"} + ADAPTER
    out = ROOT / "submission" / "my_agent_core.py"
    out.write_text(bundle)
    compile(bundle, str(out), "exec")
    print(f"wrote {out} ({len(bundle.splitlines())} lines, {len(bundle)//1024} KB)")
    return out


def check(out, game="tn36", max_actions=200):
    """Play a game offline THROUGH THE ADAPTER (the harness loop
    mimicked), in a clean namespace with no repo on the path."""
    import importlib.util, runpy
    for k in [k for k in sys.modules if k.startswith(("agent", "tools"))]:
        del sys.modules[k]
    saved = list(sys.path)
    sys.path = [p for p in sys.path if p not in (str(ROOT), "", str(ROOT / "tools"))]
    ns = runpy.run_path(str(out), run_name="bundle")
    sys.path = saved
    import arc_agi
    from arcengine import GameAction, GameState
    arc = arc_agi.Arcade(operation_mode="offline")
    gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(game)][0]
    env = arc.make(gid)
    obs = env.reset()
    ag = ns["MyAgent"]()
    frames = [obs]
    n = 0
    while n < max_actions and not ag.is_done(frames, obs):
        act = ag.choose_action(frames, obs)
        data = act.action_data.model_dump() if act.is_complex() else None
        obs = env.step(act, data=data)
        frames.append(obs); n += 1
        if obs.state is GameState.WIN:
            break
    print(f"check {game}: {n} actions, levels {obs.levels_completed}, state {obs.state}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="?", const="tn36", default=None)
    ap.add_argument("--max-actions", type=int, default=200)
    a = ap.parse_args()
    out = build()
    if a.check:
        check(out, a.check, a.max_actions)
