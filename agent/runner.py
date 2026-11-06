"""Episode runner: plays one game with a given agent, logs every transition."""
import json
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction, GameState


class EpisodeLog:
    def __init__(self, out_dir: str, game_id: str, agent_name: str):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(out_dir) / f"{game_id}__{agent_name}__{ts}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def write(self, record: dict):
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def frame_to_list(obs):
    return np.array(obs.frame).tolist()


def run_episode(arc, game_id: str, agent, max_actions: int = 2000,
                log_dir: str = "logs", log_frames: bool = False):
    """Play one scorecard-tracked episode. Returns summary dict."""
    env = arc.make(game_id)
    obs = env.reset()
    log = EpisodeLog(log_dir, game_id, agent.name)
    agent.on_episode_start(obs)

    actions_taken = 0
    per_level_actions = {}  # level index -> actions used to clear it
    actions_this_level = 0
    last_levels = obs.levels_completed

    while actions_taken < max_actions and obs.state not in (GameState.WIN,):
        action = agent.act(obs)
        data = None
        if getattr(action, "is_complex", None) and action.is_complex():
            data = action.action_data.model_dump()
        prev = np.array(obs.frame)
        obs = env.step(action, data=data)
        actions_taken += 1
        actions_this_level += 1
        changed = not np.array_equal(prev, np.array(obs.frame))
        agent.observe(action, obs, changed)

        rec = {
            "n": actions_taken,
            "action": action.name if hasattr(action, "name") else str(action),
            "changed": bool(changed),
            "state": obs.state.name,
            "levels": obs.levels_completed,
        }
        if log_frames:
            rec["frame"] = frame_to_list(obs)
        log.write(rec)

        if obs.levels_completed > last_levels:
            per_level_actions[obs.levels_completed] = actions_this_level
            actions_this_level = 0
            last_levels = obs.levels_completed
            agent.on_level_up(obs)

        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET)
            actions_taken += 1  # RESET counts as an action in scoring
            if hasattr(agent, "on_game_over"):
                agent.on_game_over(obs)

    log.close()
    return {
        "game_id": game_id,
        "agent": agent.name,
        "actions": actions_taken,
        "levels_completed": obs.levels_completed,
        "win_levels": obs.win_levels,
        "won": obs.state == GameState.WIN,
        "per_level_actions": per_level_actions,
        "log": str(log.path),
    }
