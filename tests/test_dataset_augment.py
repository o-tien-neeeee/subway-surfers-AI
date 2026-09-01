"""Tests for the augmented :class:`DemonstrationDataset`.

Covers:
* keypress-window selection applied to the index,
* horizontal mirror producing the right actions and a +1 in mirror
  count, without changing human press counts,
* split_by_episode keeping an episode and its mirror in the same fold,
* legacy datasets (no augmentor passed) still work.
"""

from __future__ import annotations

import numpy as np

from config import NOOP, LEFT, RIGHT, JUMP, SLIDE
from dataset import DemonstrationDataset, Episode
from demo_augment import DemoAugmentor


def _make_episode(actions: list[int], n: int = None) -> Episode:
    """Build a synthetic episode with the given action sequence.

    Frames are arbitrary 84x84 uint8 — the dataset does not look at the
    pixels, only at the action column and the augmentation rule.
    """
    a = np.asarray(actions, dtype=np.int64)
    n = n or len(actions)
    if n != len(actions):
        # If a different size was requested, pad/truncate (tests that
        # do not care about length just want matching shapes).
        a = a[:n]
        if len(a) < n:
            a = np.concatenate([a, np.zeros(n - len(a), dtype=np.int64)])
    frames = np.zeros((n, 84, 84), dtype=np.uint8)
    # Tag every frame with the action id at the brightest column so a
    # test could, if it ever wanted to, check the pixels too.
    for i, ai in enumerate(a):
        frames[i, :, int(ai) * 20] = 255
    return Episode(
        path="<synthetic>",
        frames=frames,
        actions=a,
        timestamps=np.arange(n, dtype=np.float64) / 30.0,
        done=np.zeros(n, dtype=bool),
    )


class TestAugmentedDataset:
    def test_default_keeps_only_keypress_windows(self) -> None:
        ep = _make_episode(
            [NOOP, NOOP, NOOP, NOOP, LEFT, NOOP, NOOP, NOOP, NOOP, NOOP]
        )
        ds = DemonstrationDataset([ep], stack=4,
                                  augment=DemoAugmentor(
                                      keypress_window=True,
                                      keypress_pre=2, keypress_post=2,
                                      mirror_horizontal=False))
        # With pre=2 and post=2 around the LEFT at index 4, the kept
        # window is 2..6 — 5 frames.  Mirror is off so no extra rows.
        assert len(ds) == 5

    def test_mirror_doubles_index_and_swaps_lanes(self) -> None:
        ep = _make_episode(
            [NOOP, LEFT, NOOP, NOOP, NOOP, NOOP]  # only one press at idx 1
        )
        ds = DemonstrationDataset(
            [ep], stack=4,
            augment=DemoAugmentor(keypress_pre=1, keypress_post=1,
                                  mirror_horizontal=True),
        )
        # Kept window around the press: 0..2 (3 frames).  Mirror adds
        # 3 more -> 6 entries total.
        assert len(ds) == 6
        counts = ds.class_counts()
        # The original kept frames: NOOP, LEFT, NOOP -> 2 NOOPs, 1 LEFT.
        # The mirror copy: NOOP, RIGHT, NOOP -> +1 RIGHT, +1 NOOP.
        assert counts[NOOP] == 4
        assert counts[LEFT] == 1
        assert counts[RIGHT] == 1

    def test_press_count_is_human_only(self) -> None:
        ep = _make_episode(
            [NOOP, NOOP, LEFT, NOOP, NOOP, NOOP]  # one press at idx 2
        )
        ds = DemonstrationDataset([ep], stack=4)
        # Mirror adds 1 RIGHT; the press-count for LEFT must still be 1.
        assert ds.dodge_press_counts()[LEFT] == 1
        assert ds.dodge_press_counts().get(RIGHT, 0) == 0
        # The mirror report lives in a separate counter.
        assert ds.mirror_added_counts()[RIGHT] == 1

    def test_oversample_multiplies_count_not_presses(self) -> None:
        ep = _make_episode(
            [NOOP, NOOP, LEFT, NOOP, NOOP, NOOP]
        )
        ds = DemonstrationDataset(
            [ep], stack=4, dodge_oversample=3,
            augment=DemoAugmentor(keypress_pre=2, keypress_post=2,
                                  mirror_horizontal=False),
        )
        # Keypress window is 0..4 (pre=2 post=2 around idx 2) = 5 frames.
        # Oversample 3x ONLY for the actual press frame (idx 2) — the
        # rest are NOOP rows that stay at x1 because they are not the
        # life-critical signal.  So: 4 NOOP x1 + 1 LEFT x3 = 7 entries.
        assert len(ds) == 7
        assert ds.dodge_press_counts()[LEFT] == 1
        # Class counts on the augmented index confirm the same split:
        # 4 NOOPs + 3 LEFTs (oversampled).
        counts = ds.class_counts()
        assert counts[NOOP] == 4
        assert counts[LEFT] == 3

    def test_split_keeps_episode_and_mirror_together(self) -> None:
        # Two episodes, each with a press; mirror is on so each press
        # has both LEFT and RIGHT copies.  The split must put the WHOLE
        # episode (original + mirror) in the same fold — otherwise val
        # accuracy measures memorisation, not generalisation.
        ep_a = _make_episode([NOOP, LEFT, NOOP, NOOP])
        ep_b = _make_episode([NOOP, RIGHT, NOOP, NOOP])
        ds = DemonstrationDataset([ep_a, ep_b], stack=4)
        train, val = ds.split_by_episode(val_fraction=0.5, seed=0)
        # Each episode has the same number of kept+mirror entries; the
        # split is by episode so val cannot be the mirror of a train
        # episode (val=one whole episode, train=the other whole one).
        train_eps = {self._entry_ep(ds, i) for i in train}
        val_eps = {self._entry_ep(ds, i) for i in val}
        assert train_eps.isdisjoint(val_eps)
        assert train_eps | val_eps == {0, 1}

    @staticmethod
    def _entry_ep(ds: DemonstrationDataset, i: int) -> int:
        return ds._index[i][0]

    def test_no_presses_in_episode(self) -> None:
        ep = _make_episode([NOOP, NOOP, NOOP, NOOP])
        ds = DemonstrationDataset([ep], stack=4)
        assert len(ds) == 0
        assert ds.dodge_press_counts() == {a: 0 for a in range(5)}

    def test_get_returns_mirrored_action(self) -> None:
        ep = _make_episode([NOOP, LEFT, NOOP])
        ds = DemonstrationDataset(
            [ep], stack=4,
            augment=DemoAugmentor(keypress_pre=1, keypress_post=1,
                                  mirror_horizontal=True),
        )
        # Find the mirrored LEFT entry.
        found_orig = found_mirror = None
        for i in range(len(ds)):
            ei, si, m = ds._index[i]
            if ei == 0 and si == 1:
                stack, action = ds.get(i)
                if m == 0:
                    found_orig = action
                else:
                    found_mirror = action
        assert found_orig == LEFT
        assert found_mirror == RIGHT
        # The mirrored stack must be a horizontal flip of the original.
        # The helper placed a column marker at column == action * 20
        # (LEFT -> col 20).  After a full horizontal mirror, that column
        # is at W - 1 - 20 = 84 - 1 - 20 = 63.  Both have to be in the
        # same stack slot (newest), proving stack_mirror is the default.
        for i in range(len(ds)):
            ei, si, m = ds._index[i]
            if ei == 0 and si == 1 and m == 1:
                stack, _ = ds.get(i)
                col = int(np.argmax(stack[-1].sum(axis=0)))
                assert col == 63  # 84 - 1 - 20

    def test_class_weights_still_well_defined(self) -> None:
        ep = _make_episode([NOOP, NOOP, LEFT, NOOP, RIGHT, NOOP, JUMP, NOOP])
        ds = DemonstrationDataset([ep], stack=4)
        w = ds.class_weights("inverse_sqrt")
        # All weights in [0, 1] and finite — required by the loss path.
        assert w.shape == (5,)
        assert np.all(np.isfinite(w))
        assert (w >= 0).all() and (w <= 1).all()

    def test_legacy_no_augmentor_argument(self) -> None:
        # Passing no augmentor defaults to DemoAugmentor() with the
        # keypress-window + mirror BOTH on.  The legacy "every frame
        # mode" must be opted into explicitly.
        ep = _make_episode([NOOP, LEFT, NOOP, RIGHT, NOOP])
        ds = DemonstrationDataset([ep], stack=4)
        # Default pre=5, post=5 -> windows around idx 1 and idx 3 merge
        # to 0..4 (5 frames).  Mirror adds 5 more -> 10 entries.
        assert len(ds) == 10
        assert ds.dodge_press_counts()[LEFT] == 1
        assert ds.dodge_press_counts()[RIGHT] == 1
        # The mirror copy turns LEFT into RIGHT and vice-versa.
        assert ds.mirror_added_counts().get(RIGHT, 0) == 1
        assert ds.mirror_added_counts().get(LEFT, 0) == 1

    def test_legacy_explicit_disabled(self) -> None:
        ep = _make_episode([NOOP, LEFT, NOOP, RIGHT, NOOP])
        ds = DemonstrationDataset(
            [ep], stack=4,
            augment=DemoAugmentor(mirror_horizontal=False,
                                  keypress_window=False),
        )
        assert len(ds) == 5
        assert sum(ds.mirror_added_counts().values()) == 0
