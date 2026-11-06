"""Baseline agents.

Agent protocol:
  name: str
  on_episode_start(obs), act(obs) -> GameAction, observe(action, obs, changed), on_level_up(obs)
"""
import random
from collections import defaultdict

import numpy as np
from arcengine import GameAction

ACTION_BY_ID = {a.value: a for a in GameAction}


def available(obs):
    """Map available_actions ints -> GameAction members (excluding RESET)."""
    acts = []
    for i in obs.available_actions or []:
        a = ACTION_BY_ID.get(i)
        if a is not None and a != GameAction.RESET:
            acts.append(a)
    return acts or [GameAction.ACTION1]


class RandomAgent:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def on_episode_start(self, obs): ...
    def on_level_up(self, obs): ...
    def observe(self, action, obs, changed): ...

    def act(self, obs):
        acts = available(obs)
        a = self.rng.choice(acts)
        if a == GameAction.ACTION6:
            a.set_data({"x": self.rng.randrange(64), "y": self.rng.randrange(64)})
        return a


class NoveltyAgent:
    """Prefers actions that have historically changed the frame from the
    current (hashed) state, epsilon-greedy otherwise. A trivial stand-in for a
    real exploration policy, to exercise the harness."""
    name = "novelty"

    def __init__(self, seed=0, eps=0.25):
        self.rng = random.Random(seed)
        self.eps = eps
        self.effect = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # state -> action -> [changed, tried]
        self._last_state = None
        self._last_action = None

    def _key(self, obs):
        return hash(np.array(obs.frame).tobytes())

    def on_episode_start(self, obs):
        self._last_state = self._key(obs)

    def on_level_up(self, obs): ...

    def observe(self, action, obs, changed):
        st = self._last_state
        stats = self.effect[st][action.name]
        stats[1] += 1
        stats[0] += int(changed)
        self._last_state = self._key(obs)

    def act(self, obs):
        acts = available(obs)
        st = self._key(obs)
        self._last_state = st
        if self.rng.random() < self.eps:
            a = self.rng.choice(acts)
        else:
            def score(a):
                c, t = self.effect[st][a.name]
                return (c + 1) / (t + 2)  # optimistic: untried actions look good
            a = max(acts, key=score)
        if a == GameAction.ACTION6:
            a.set_data({"x": self.rng.randrange(64), "y": self.rng.randrange(64)})
        return a
