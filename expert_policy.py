"""Expert policy: the optimal action for the LearnableEnv.

This is the "knowing the answer" baseline that we use to
generate expert demonstrations for behaviour cloning.  It
encodes a single rule: if your current lane is the one the
next obstacle targets, move out; otherwise stay.

Why this exists
----------------
The user asked for the AI to *actually* learn.  Behaviour
cloning from an expert policy is the fastest way to seed a
working policy: a small dataset of (obs, action) pairs
from a policy that achieves the user's KPI is enough to
get a BC-pretrained model to a usable level, and from
there online RL can refine.  The expert here is *not* the
final goal — it is a launchpad.

Real-game note
--------------
For the *real* Subway Surfers game the expert would be the
user's own gameplay recorded via the demo recorder.  This
file provides the LearnableEnv-side expert so the audit
and the unit tests can pin the BC pipeline end-to-end; the
real-game expert is the human demos already supported by
``demonstration_recorder.py``.

The class is also intentionally *side-effect free*: a single
``act(obs) -> int`` call given an observation.  This makes
it easy to call from a Python loop, from a recorder, or
from a unit test that just wants to assert "this observation
should yield this action".
"""

from __future__ import annotations

import numpy as np

from learnable_env import LearnableEnv


class ExpertPolicy:
    """Optimal policy for the LearnableEnv.

    Given the 7-dim observation
    (player_one_hot + next_obstacle_one_hot + time_to_impact),
    pick the action that puts the player in a safe lane
    *before* the obstacle impacts.  If already safe, NOOP
    (which is the cheapest correct action in this env).
    """

    def __init__(self) -> None:
        self.last_player_lane = 1
        self.next_obstacle_lane = 0

    def act(self, obs: np.ndarray) -> int:
        player_lane = int(np.argmax(obs[:3]))
        next_lane = int(np.argmax(obs[3:6]))
        tti = float(obs[6])
        # If the next obstacle is in another lane or there
        # is plenty of time, just NOOP.
        if player_lane != next_lane or tti > 0.7:
            return 0
        # Otherwise move out.  Prefer LEFT, unless we're
        # already at 0, in which case RIGHT.
        if player_lane > 0:
            return 1
        return 2

    def collect_demonstration(self, env: LearnableEnv,
                                max_steps: int = 900
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Roll out the policy and return (obs, action, next_obs) triples.

        The triples are in the same format as the human demo
        recorder (``DemonstrationRecorder``): one transition
        per frame, with the action the expert chose.

        Returns
        -------
        obs : ``[T, 7]`` float32 — the *state* the policy saw.
        actions : ``[T]`` int64 — the action chosen.
        next_obs : ``[T, 7]`` float32 — the state that
            resulted from the action.
        """
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        next_buf: list[np.ndarray] = []
        obs = env.reset()
        for _ in range(max_steps):
            a = self.act(obs)
            next_obs, _, done, _ = env.step(a)
            obs_buf.append(obs)
            act_buf.append(a)
            next_buf.append(next_obs)
            obs = next_obs
            if done:
                break
        return (np.stack(obs_buf, axis=0).astype(np.float32),
                np.asarray(act_buf, dtype=np.int64),
                np.stack(next_buf, axis=0).astype(np.float32))
