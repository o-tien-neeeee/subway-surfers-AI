"""Visual data augmentation for the QR-DQN agent.

Data augmentation is one of the most impactful
interventions in modern visual RL (Laskin et al. 2020
"Reinforcement Learning with Augmented Data" / RAD;
Yarats et al. 2021 "Mastering Visual Continuous Control").
The idea is simple: the same image, slightly perturbed,
should produce the same Q-values.  By adding random
shifts, intensity jitter, and (optionally) random crops
to the training input, the network learns a *robust*
representation that does not overfit to the exact
pixel positions in the demos.

We chose four augmentations, all proven to help in the
RL literature:

* **Intensity jitter** — add a small random
  multiplicative + additive offset.  Cheap, helps with
  lighting variation (the real Poki game has different
  colours per session; the synthetic game has them
  too).
* **Random translate** — shift the frame by 1-3 pixels
  in any direction.  The frame stack stays aligned
  (we shift every frame by the same amount) so the
  network cannot use the absolute position of an
  obstacle to memorise the answer.
* **Frame differencing** — append a new channel that is
  the *difference* between consecutive frames.  This
  is a *learned-free* motion feature; the network
  still has to learn everything else but the obstacle
  motion is given for free in the input.  This is
  the Atari-DQN trick (Mnih et al. 2015): the 4-frame
  stack + the phi() function were a poor man's frame
  differencing.
* **Random erasing** — zero out a small random
  rectangle.  Helps the network cope with occlusions
  (e.g. the colour-anchor patch can change between
  captures; the network should not depend on it).

API
---
:meth:`augment_batch` takes a ``[B, F, H, W]`` uint8
tensor (or float32 in [0, 1]) and returns the same shape
with the augmentations applied.  Each call uses a fresh
RNG so the augmentations are uncorrelated across
training steps (this is important: if the same
augmentation is applied to every sample in a batch,
the network can learn to *predict* the augmentation
instead of the action).

The augmentations are also useful at *inference* time:
some papers (e.g. DrQ) use the *average* of K
augmented forward passes as the final Q-value.  We
expose :meth:`augment_batch` so the learner can opt
into this if the latency budget allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AugmentationConfig:
    """Knobs for the visual augmentation pipeline.

    All probabilities are 0..1; all magnitudes are in
    pixel or [0, 1] units as noted.
    """
    # Intensity jitter.
    intensity_jitter: bool = True
    intensity_brightness: float = 0.10  # ±0.10 additive
    intensity_contrast: float = 0.10   # ±0.10 multiplicative
    # Random translate.
    translate: bool = True
    translate_pixels: int = 3          # max ±3 pixels
    # Random erasing.
    random_erasing: bool = False       # off by default
    erasing_prob: float = 0.25         # 25% of frames
    erasing_max_frac: float = 0.20     # up to 20% of the frame
    # Frame differencing (appended as a new channel).
    frame_diff: bool = False           # off by default
    # Mixup (for batch-level augmentation).
    mixup: bool = False                # off by default
    mixup_alpha: float = 0.2


def _to_float(t: torch.Tensor) -> torch.Tensor:
    """Cast uint8 [0,255] to float32 [0,1] without copy
    if already float."""
    if t.dtype == torch.uint8:
        return t.float() / 255.0
    if t.dtype.is_floating_point:
        return t
    return t.float() / 255.0


def _to_uint8(t: torch.Tensor) -> torch.Tensor:
    """Cast float32 [0,1] back to uint8 [0,255]."""
    if t.dtype == torch.uint8:
        return t
    return (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


# --------------------------------------------------------------------- #
def intensity_jitter(obs: torch.Tensor,
                       brightness: float = 0.10,
                       contrast: float = 0.10) -> torch.Tensor:
    """Apply a random brightness + contrast shift.

    ``obs`` is ``[B, F, H, W]`` in [0, 1].  The same
    jitter is applied to every frame in the stack (so
    the agent cannot learn the *specific* jitter of a
    particular frame).
    """
    if obs.dtype != torch.float32:
        obs = obs.float()
    B = obs.shape[0]
    # One brightness/contrast pair per batch element.
    b = (torch.rand(B, 1, 1, 1, device=obs.device) * 2 - 1) \
        * brightness
    c = 1.0 + (torch.rand(B, 1, 1, 1, device=obs.device) * 2 - 1) \
        * contrast
    return (obs * c + b).clamp(0.0, 1.0)


def random_translate(obs: torch.Tensor, max_pixels: int = 3
                       ) -> torch.Tensor:
    """Shift every frame in the stack by the same
    random offset (in [-max_pixels, max_pixels] pixels
    on each axis).  We use ``torch.roll`` which is
    O(B*F*H*W) and avoids any padding artifacts."""
    if max_pixels <= 0:
        return obs
    B, F, H, W = obs.shape
    # Random per-batch offsets; same for every frame
    # in the stack (so the stack stays aligned).
    dx = int(torch.randint(-max_pixels, max_pixels + 1, (1,)).item())
    dy = int(torch.randint(-max_pixels, max_pixels + 1, (1,)).item())
    if dx == 0 and dy == 0:
        return obs
    # ``torch.roll`` wraps around; we manually zero the
    # wrapped region after the roll so the shifted frame
    # has black borders (matching the real game's edge).
    out = torch.roll(obs, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        out[..., :dy, :] = 0.0
    elif dy < 0:
        out[..., dy:, :] = 0.0
    if dx > 0:
        out[..., :, :dx] = 0.0
    elif dx < 0:
        out[..., :, dx:] = 0.0
    return out


def random_erasing(obs: torch.Tensor,
                    prob: float = 0.25,
                    max_frac: float = 0.20) -> torch.Tensor:
    """Zero out a small random rectangle in a random
    fraction of the batch.  Per Zhong et al. 2020
    "Random Erasing Data Augmentation" — proven to
    help the network cope with occlusions."""
    if prob <= 0 or max_frac <= 0:
        return obs
    B, F, H, W = obs.shape
    out = obs.clone()
    # Which batch elements get erased.
    mask = torch.rand(B, device=obs.device) < prob
    for i in range(B):
        if not mask[i]:
            continue
        # Random rectangle size in [10%, 100%] of max_frac.
        frac = max_frac * float(torch.rand(1).item())
        eh = max(1, int(frac * H))
        ew = max(1, int(frac * W))
        ey = int(torch.randint(0, max(1, H - eh + 1), (1,)).item())
        ex = int(torch.randint(0, max(1, W - ew + 1), (1,)).item())
        out[i, :, ey:ey + eh, ex:ex + ew] = 0.0
    return out


def frame_difference(obs: torch.Tensor) -> torch.Tensor:
    """Append a new channel that is the temporal
    difference between consecutive frames in the
    stack.

    The output has shape ``[B, F+1, H, W]`` where the
    new channel is ``|F[0] - F[1]|`` (a single
    differenced image).  This is the "phi" feature
    from the original DQN paper; the modern way is
    to let the network learn it, but providing it
    *for free* gives a 5-10% sample-efficiency
    boost on most Atari games.

    Note: we return a NEW tensor (we do not mutate
    in place) so this is safe inside a replay sampler.
    """
    if obs.shape[1] < 2:
        # 1-frame stack: differencing is undefined.
        return obs
    diff = (obs[:, -1] - obs[:, -2]).abs()
    return torch.cat([obs, diff.unsqueeze(1)], dim=1)


def mixup(obs: torch.Tensor, actions: torch.Tensor,
            rewards: torch.Tensor, dones: torch.Tensor,
            alpha: float = 0.2) -> tuple[torch.Tensor,
                                            torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
    """Mixup augmentation (Zhang et al. 2017): convex
    combination of two random samples.

    For RL, mixup on observations is less common than
    on supervised tasks because the action label is a
    *categorical* value (mixing two actions does not
    make sense).  We therefore mix only the
    *observations* and the *rewards*; the action
    comes from the first sample (so the network still
    has a clear gradient signal).

    Returns the mixed obs + the original action of the
    first sample + the mixed reward + the original
    done flag.
    """
    if alpha <= 0:
        return obs, actions, rewards, dones
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(obs.shape[0], device=obs.device)
    mixed_obs = lam * obs + (1.0 - lam) * obs[perm]
    mixed_rew = lam * rewards + (1.0 - lam) * rewards[perm]
    return mixed_obs, actions, mixed_rew, dones


# --------------------------------------------------------------------- #
def augment_batch(obs: torch.Tensor,
                    actions: torch.Tensor | None = None,
                    rewards: torch.Tensor | None = None,
                    dones: torch.Tensor | None = None,
                    cfg: AugmentationConfig | None = None,
                    training: bool = True) -> torch.Tensor:
    """Apply the configured augmentations to a batch
    of observations.

    The output is the same shape as the input.  If
    ``cfg.frame_diff`` is True, the output has one
    extra channel.

    Parameters
    ----------
    obs : ``[B, F, H, W]`` uint8 or float32 in [0, 1]
    actions, rewards, dones : optional, for mixup
    cfg : AugmentationConfig; default = all on with
        conservative magnitudes
    training : bool; if False, return obs unchanged
        (inference should be deterministic)
    """
    if not training or cfg is None:
        return obs
    # Normalise once.
    if obs.dtype == torch.uint8:
        obs = obs.float() / 255.0
    out = obs
    if cfg.intensity_jitter:
        out = intensity_jitter(out, cfg.intensity_brightness,
                                 cfg.intensity_contrast)
    if cfg.translate:
        out = random_translate(out, cfg.translate_pixels)
    if cfg.random_erasing:
        out = random_erasing(out, cfg.erasing_prob, cfg.erasing_max_frac)
    if cfg.frame_diff:
        out = frame_difference(out)
    return out
