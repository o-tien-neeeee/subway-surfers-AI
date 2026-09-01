"""Expert policy for the (random-obstacle) ``SyntheticGame``.

The :class:`expert_policy.ExpertPolicy` in ``expert_policy.py``
is the optimal policy for :class:`learnable_env.LearnableEnv`
which has a *deterministic* obstacle schedule (the audit
benchmark).  The real Subway Surfers game is closer to
:class:`environment.SyntheticGame`: obstacles spawn
randomly (lane + kind) on a Poisson-ish timer, and the
agent must react to whatever comes.

This module provides :class:`SyntheticExpert`, the
"knowing the answer" policy for ``SyntheticGame``:

* It looks at the **next** obstacle that is still on the
  track (the one closest to the player that has not
  reached ``prog=1.0`` yet).
* It picks the dodge action that puts the player in a
  *safe* lane — the one without an obstacle in the
  collision row.
* For ``"low"`` barriers the safe action is JUMP; for
  ``"high"`` barriers the safe action is SLIDE.

Because the obstacle schedule is random, the expert's
accuracy is bounded by the *lookahead*: if a new obstacle
spawns after the expert's decision it has to dodge it
with a brand-new decision.  The expert still
*out-performs random play* by a wide margin, which is
what makes the BC pretrain useful (the agent learns
"if I see a low barrier, jump" instead of having to
figure that out from scratch).

API parity
----------
:class:`SyntheticExpert.act` and
:meth:`SyntheticExpert.collect_demonstration` mirror the
:class:`expert_policy.ExpertPolicy` API so the BC pretrain
module can be called with either env.  The key
difference is that the SyntheticGame expert takes the
**game state** (player_lane, obstacles list) as input,
not a learned embedding, because there is no pre-defined
observation vector for SyntheticGame.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Action constants (must match config.ACTIONS).
NOOP = 0
LEFT = 1
RIGHT = 2
JUMP = 3
SLIDE = 4


class SyntheticExpert:
    """The optimal policy for ``SyntheticGame``.

    The expert is given the *game state* (player_lane
    + the list of active obstacles) and returns the
    best action.  "Best" means:

    * If the next obstacle is a ``"low"`` barrier in
      the player's lane → JUMP.
    * If the next obstacle is a ``"high"`` barrier in
      the player's lane → SLIDE.
    * If the next obstacle is a ``"lane"`` blocker in
      the player's lane → move to the *nearest* safe
      lane (left or right, preferring the side with
      more room).
    * Otherwise → NOOP (no immediate threat).

    A small look-ahead buffer keeps track of the next
    few obstacles so the expert can choose LEFT vs
    RIGHT based on the obstacle *behind* the immediate
    one (the player does not want to dodge into
    another obstacle).
    """

    def __init__(self, lookback: int = 4) -> None:
        self.lookback = lookback
        # Track the most recent player lane so the
        # recorder can detect "no movement" frames and
        # log them (a real "instant death" diagnostic
        # on the real game would be similar).
        self._last_lane: int | None = None

    # ------------------------------------------------------------------ #
    def act(self, player_lane: int, obstacles: list[dict]) -> int:
        """Pick the optimal action for the current
        game state.

        Parameters
        ----------
        player_lane : int
            ``0``, ``1`` or ``2`` — the lane the player
            is currently in.
        obstacles : list[dict]
            The active obstacle list from
            ``SyntheticGame.obstacles`` (each dict has
            keys ``kind``, ``lane``, ``prog``, ``speed``).

        Returns
        -------
        action : int
            One of ``NOOP/LEFT/RIGHT/JUMP/SLIDE``.

        Strategy
        --------
        The expert works in two stages:

        1. **Immediate threat**: if any obstacle in
           the player's lane is at ``prog >= 0.05``,
           we must act on it NOW.  A low barrier
           needs JUMP, a high barrier needs SLIDE,
           and a lane blocker needs to be dodged
           (left/right).
        2. **Pre-emptive move**: if there is no
           immediate threat, but a lane-blocker is
           approaching at high prog (>= 0.4) and the
           player is on a lane that will be blocked,
           we move EARLY — to a safe lane — so the
           player is not caught in mid-dodge when
           the obstacle hits.  This is the difference
           between "8s expert" and "20s expert" we
           measured on the first iteration.
        """
        # Sort by progress so the closest threat is
        # at the front of the list.
        live = [ob for ob in obstacles
                if ob["prog"] < 0.95]
        if not live:
            return NOOP
        live.sort(key=lambda ob: -ob["prog"])
        # Stage 1: immediate threat.
        for ob in live:
            if ob["lane"] != player_lane:
                continue
            kind = ob["kind"]
            if kind == "low":
                return JUMP
            if kind == "high":
                return SLIDE
            if kind == "lane":
                # Imminent blocker.  Pick a safe lane
                # (one without a blocker in the next
                # few obstacles) and move.
                safe = self._safe_lanes(player_lane, live,
                                          lookback=self.lookback)
                if not safe:
                    # No fully-safe lane (rare).  Move
                    # to the lane with the *most* lead
                    # time before the next blocker.
                    return self._best_of_bad_lanes(
                        player_lane, live)
                target = min(safe,
                              key=lambda l: abs(l - player_lane))
                if target < player_lane:
                    return LEFT
                if target > player_lane:
                    return RIGHT
                return NOOP
        # Stage 2: pre-emptive move.  If a blocker is
        # coming at high prog and we're on its lane,
        # move NOW so we don't have to react in the
        # last 30% before impact.
        for ob in live:
            if ob["kind"] != "lane":
                continue
            if ob["prog"] < 0.4:
                # Too far away to pre-empt.
                break
            if ob["lane"] != player_lane:
                continue
            # About to be blocked; pre-empt.
            safe = self._safe_lanes(player_lane, live,
                                      lookback=self.lookback)
            if not safe or player_lane in safe:
                # Player is already on a safe lane OR
                # every lane is contested (rare).
                # Don't bother.
                return NOOP
            target = min(safe,
                          key=lambda l: abs(l - player_lane))
            if target < player_lane:
                return LEFT
            if target > player_lane:
                return RIGHT
        return NOOP

    def _best_of_bad_lanes(self, player_lane: int,
                            live: list[dict]) -> int:
        """When no lane is safe, pick the lane with
        the *most* lead time before the next
        blocker.  This is a soft preference — the
        agent might still die, but the move at
        least buys time."""
        # Compute the lead time (1 - prog) for the
        # next blocker in each lane.
        lead: dict[int, float] = {0: 1.0, 1: 1.0, 2: 1.0}
        for ob in live:
            if ob["kind"] != "lane":
                continue
            lead[ob["lane"]] = min(lead[ob["lane"]],
                                       1.0 - ob["prog"])
        # Pick the lane with the highest lead, tie-
        # broken by smallest distance from the
        # player's current lane.
        best = max((0, 1, 2),
                    key=lambda l: (lead[l], -abs(l - player_lane)))
        if best < player_lane:
            return LEFT
        if best > player_lane:
            return RIGHT
        return NOOP

    # ------------------------------------------------------------------ #
    def _safe_lanes(self, player_lane: int,
                     immediate: list[dict],
                     lookback: int) -> list[int]:
        """Return the list of lanes that have no
        upcoming ``"lane"`` obstacle in the next
        ``lookback`` obstacles (i.e. the next 1-2
        steps)."""
        upcoming_lanes: set[int] = set()
        for ob in immediate[:lookback]:
            if ob["kind"] == "lane":
                upcoming_lanes.add(ob["lane"])
        return [lane for lane in (0, 1, 2)
                if lane not in upcoming_lanes]

    # ------------------------------------------------------------------ #
    def collect_demonstration(self, env, max_steps: int = 900,
                                ) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray]:
        """Roll the expert out on a ``SyntheticGame``
        and return (frames, actions, rewards).

        The frame stack is a 4-frame 84x84 grayscale
        stack — the same observation the production
        agent sees.  We render and preprocess each
        frame so the BC dataset is directly usable by
        :class:`dataset.DemonstrationDataset`.

        Returns
        -------
        frames : ``[T, 4, 84, 84]`` uint8
        actions : ``[T]`` int64
        rewards : ``[T]`` float32
        """
        from PIL import Image

        env.reset()
        # Seed the frame stack with the first 4 copies
        # of the initial frame.
        f0 = env.render()
        gray0 = self._preprocess(f0)
        stack = [gray0] * 4
        frames_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        rew_buf: list[float] = []
        for _ in range(max_steps):
            obs = np.stack(stack, axis=0)
            a = self.act(env.player_lane, env.obstacles)
            step = env.step_with_reward(a)
            stack = stack[1:] + [self._preprocess(step["frame"])]
            frames_buf.append(obs)
            act_buf.append(int(a))
            rew_buf.append(float(step["reward"]))
            if step["done"]:
                break
        if not frames_buf:
            return (np.zeros((0, 4, 84, 84), dtype=np.uint8),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.float32))
        return (np.stack(frames_buf, axis=0),
                np.asarray(act_buf, dtype=np.int64),
                np.asarray(rew_buf, dtype=np.float32))

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) -> 84x84 uint8 grayscale."""
        gray = frame.mean(axis=2).astype(np.uint8)
        img = Image.fromarray(gray).resize((84, 84))
        return np.asarray(img, dtype=np.uint8)
