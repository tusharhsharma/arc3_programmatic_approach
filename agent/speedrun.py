"""Two-run competition architecture (verified optimal under official scoring:
per-game score = MAX over runs; see probes #1/#2, 2026-08-27).

Phase 1 — LEARNING RUN (free): MapperAgent explores, dies, resets at will.
A recorder keeps, per level, the action sequence of the FINAL SUCCESSFUL
attempt (a death or reset wipes the in-progress segment: the level restarts,
so only the attempt that actually completed the level is kept).

Phase 2 — SPEED-RUN (scored): fresh environment; replay each level's pruned
sequence (actions that changed nothing are dropped). Any divergence — game
over, exhausted segment without level-up — degrades gracefully to the live
agent, which still carries its own learning. Under max-scoring a failed
speed-run costs nothing; a good one sets the score.
"""
from arcengine import GameAction

from .mapper_agent import MapperAgent


class PlaybookRecorder:
    """Attach to a learning run; produces {level_number: [(name, data, changed)]}."""

    def __init__(self):
        self.playbook = {}
        self._segment = []
        self._level = 0

    def start(self, obs):
        self._level = obs.levels_completed
        self._segment = []

    def record(self, action, data, changed, obs):
        name = action.name if hasattr(action, "name") else str(action)
        if name == "RESET":
            self._segment = []          # level restarted: attempt discarded
            return
        self._segment.append((name, dict(data) if data else None, bool(changed)))
        if obs.levels_completed > self._level:
            # this attempt completed the level: freeze it
            self.playbook[self._level + 1] = list(self._segment)
            self._level = obs.levels_completed
            self._segment = []

    def on_game_over(self):
        self._segment = []              # death: attempt discarded


def prune(segment):
    """Drop actions that changed nothing (wall bumps, inert probes).
    If everything gets pruned, keep the original."""
    kept = [s for s in segment if s[2]]
    return kept if kept else list(segment)


class SpeedRunAgent:
    """Replays a playbook level by level; degrades to a live MapperAgent."""
    name = "speedrun"

    def __init__(self, playbook, seed=0, use_pruned=True):
        self.inner = MapperAgent(seed=seed)
        self.playbook = playbook or {}
        self.use_pruned = use_pruned
        self.queue = []
        self.mode = "replay"
        self._levels = 0

    # ---------- lifecycle ----------

    def on_episode_start(self, obs):
        self.inner.on_episode_start(obs)
        self._levels = obs.levels_completed
        self._load_segment()

    def on_level_up(self, obs):
        self.inner.on_level_up(obs)
        self._levels = obs.levels_completed
        self._load_segment()

    def on_game_over(self, obs):
        self.inner.on_game_over(obs)
        self.queue = []                 # alignment lost: go live
        self.mode = "live"

    def _load_segment(self):
        seg = self.playbook.get(self._levels + 1)
        if seg:
            self.queue = prune(seg) if self.use_pruned else list(seg)
            self.mode = "replay"
        else:
            self.queue = []
            self.mode = "live"          # beyond learned levels: keep learning

    # ---------- acting ----------

    def act(self, obs):
        if self.mode == "replay":
            if self.queue:
                name, data, _ = self.queue.pop(0)
                a = GameAction[name]
                if data:
                    a.set_data(data)
                return a
            # segment exhausted without a level-up: replay failed, go live
            self.mode = "live"
        return self.inner.act(obs)

    def observe(self, action, obs, changed):
        # the inner agent keeps learning even while we replay, so a fallback
        # starts warm, not blind
        self.inner.observe(action, obs, changed)
