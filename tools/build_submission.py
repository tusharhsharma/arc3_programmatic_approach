"""Bundle the multi-module agent into a single my_agent.py for the official
ARC-AGI-3 Kaggle starter kit (which splices exactly one file into the
submission notebook).

Concatenates our modules in dependency order, strips intra-package imports
(everything shares one namespace in the bundle), and appends the MyAgent
adapter implementing the framework contract:
    class MyAgent(Agent): is_done(...), choose_action(...)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "agent/perception.py",
    "agent/tracker.py",
    "agent/world_model.py",
    "agent/goals.py",
    "agent/transform_solver.py",
    "agent/docking.py",
    "agent/multibody.py",
    "agent/permuter.py",
    "agent/pegs.py",
    "agent/aligner.py",
    "agent/scene_analysis.py",
    "agent/agents.py",
    "agent/mapper_agent.py",
    "agent/speedrun.py",
]

HEADER = '''"""ARC Prize 2026 — ARC-AGI-3 submission: "The Equipped Scientist".

A fully symbolic agent (no neural networks): a library of given priors
(geometry, congruence, symmetry, object persistence), online world-model
learning (occupancy, action semantics, drift decomposition, color->terrain
generalization), hypothesis commitment with honest refutation, and
convergence-guaranteed exploration (escalating death-marks, click-sequence
ledger). Auto-generated single-file bundle — do not edit by hand; edit the
modules in the arc3 repo and re-run tools/build_submission.py.
"""
# NOTE: no `from __future__ import annotations` here — string annotations
# break dataclasses when a harness execs this file without registering it
# in sys.modules (dataclasses._is_type -> sys.modules.get(module) -> None).

import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

'''

ADAPTER = '''

# ═════════════════════════════════════════════════════════════════════════
# Framework adapter: the callback interface Kaggle's harness drives.
# ═════════════════════════════════════════════════════════════════════════

class MyAgent(Agent):
    """Equipped-scientist agent behind the ARC-AGI-3-Agents contract."""

    # The framework stops at the agent's own MAX_ACTIONS (default 80!).
    # Scoring is per-level, so extra actions on an unfinished level never
    # hurt banked level scores: a bigger cap is non-decreasing for score.
    # TIME_BUDGET_S guards the notebook wall-clock (games run as parallel
    # threads in one process; is_done=True ends this game's session).
    MAX_ACTIONS = 4000
    TIME_BUDGET_S = 100 * 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._brain = MapperAgent(seed=0)
        self._started = False
        self._prev_action = None
        self._levels_seen = 0
        self._was_game_over = False
        self._t0 = None

    def is_done(self, frames: list, latest_frame) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if self._t0 is None:
            self._t0 = time.time()
        return (time.time() - self._t0) > self.TIME_BUDGET_S

    def choose_action(self, frames: list, latest_frame) -> "GameAction":
        # bootstrap / death handling: the framework expects us to RESET
        if latest_frame.state is GameState.NOT_PLAYED:
            self._prev_action = None
            return GameAction.RESET
        if latest_frame.state is GameState.GAME_OVER:
            self._prev_action = None
            self._was_game_over = True
            return GameAction.RESET

        if not self._started:
            self._brain.on_episode_start(latest_frame)
            self._started = True
            self._levels_seen = latest_frame.levels_completed
        else:
            if self._was_game_over:
                self._brain.on_game_over(latest_frame)
                self._was_game_over = False
            elif self._prev_action is not None:
                self._brain.observe(self._prev_action, latest_frame, True)
            if latest_frame.levels_completed > self._levels_seen:
                self._levels_seen = latest_frame.levels_completed
                self._brain.on_level_up(latest_frame)

        try:
            action = self._brain.act(latest_frame)
        except Exception:
            # never crash the harness: a random legal action beats a zero
            cands = [a for a in GameAction if a is not GameAction.RESET]
            action = random.choice(cands)
            if action.is_complex():
                action.set_data({"x": random.randint(0, 63),
                                 "y": random.randint(0, 63)})
        self._prev_action = action
        return action
'''


def strip_module(src: str) -> str:
    out = []
    for line in src.splitlines():
        # only strip TOP-LEVEL imports (column 0): function-local imports
        # (e.g. `from collections import Counter as C` inside a method)
        # must survive the bundling
        if line[:1] not in (" ", "\t"):
            s = line.strip()
            if (s.startswith("from .") or s.startswith("from agents.")
                    or s.startswith("import numpy") or s == "import random"
                    or s == "import time"
                    or s.startswith("from collections import")
                    or s.startswith("from dataclasses import")
                    or s.startswith("from arcengine import")
                    or s.startswith("from typing import")
                    or s.startswith("from __future__")):
                # guard: a PARENTHESIZED multi-line import would leave its
                # continuation lines behind as garbage in the bundle
                if "(" in s and ")" not in s:
                    raise SystemExit(
                        f"MULTI-LINE IMPORT: {s!r} — the bundler strips "
                        "line-by-line; write package imports on one line")
                continue
        out.append(line)
    return "\n".join(out)


def main():
    parts = [HEADER]
    for m in MODULES:
        src = (ROOT / m).read_text()
        parts.append(f"\n# {'─'*70}\n# module: {m}\n# {'─'*70}\n")
        parts.append(strip_module(src))
    parts.append(ADAPTER)
    bundle = "\n".join(parts)
    # collision guard: a top-level name defined in two modules silently
    # shadows in the bundle (observed: tracker.unit vs world_model.unit
    # broke drift handling) — refuse to build on any duplicate
    import re as _re
    seen = {}
    for m in MODULES:
        src = (ROOT / m).read_text()
        for match in _re.finditer(r"^(?:def|class)\s+(\w+)", src, _re.M):
            name = match.group(1)
            if name in seen and seen[name] != m:
                raise SystemExit(
                    f"NAME COLLISION: '{name}' defined in {seen[name]} and {m}")
            seen[name] = m
    # guard: function-local imports of our own package survive stripping
    # and crash on Kaggle where the package doesn't exist (bit us twice)
    import re as _re2
    for m2 in MODULES:
        src2 = (ROOT / m2).read_text()
        for line in src2.splitlines():
            if line[:1] in (" ", "\t") and _re2.match(
                    r"\s+from (agent\.|\.)", line):
                raise SystemExit(
                    f"LOCAL PACKAGE IMPORT in {m2}: {line.strip()!r} — "
                    "hoist it to module level (the bundler strips those)")
    # guard: top-level aliased imports of our own modules ('import X as Y')
    # strip away and leave Y undefined in the bundle (bit us with _OPS and
    # perm_solve) — require unaliased names
    for m3 in MODULES:
        src3 = (ROOT / m3).read_text()
        for line in src3.splitlines():
            if line[:1] not in (" ", "\t") and line.startswith("from .")                     and " as " in line:
                raise SystemExit(
                    f"ALIASED PACKAGE IMPORT in {m3}: {line.strip()!r} — "
                    "the bundler strips it; use the unaliased name")
    out = ROOT / "submission" / "my_agent.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(bundle)
    print(f"wrote {out} ({len(bundle.splitlines())} lines)")
    compile(bundle, str(out), "exec")
    print("syntax OK")


if __name__ == "__main__":
    main()
