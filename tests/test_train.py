"""
Tests for train.py — PyTorch training loop components.

All tests work without real self-play files wherever possible;
temporary files are used for file I/O tests.
"""
import io
import os
import random
import struct
import sys
import tempfile

import numpy
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import twixt
import naf
from twixt import Point, Game
from naf import LearningState
from model import TwixNet
from train import (
    make_policy_target,
    make_value_target,
    FileInfo,
    load_selector,
    prepare_batch,
    Trainer,
    read_from_holdout,
)

NUM_MOVES = LearningState.NUM_MOVES  # 528
SIZE = Game.SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_learning_state(z=1, seed=0):
    """Return a LearningState with deterministic visit counts."""
    rng = numpy.random.RandomState(seed)
    g = Game()
    ls = LearningState(g)
    N = rng.randint(0, 100, NUM_MOVES).astype(numpy.uint16)
    N[0] = max(N[0], 1)   # ensure at least one non-zero count
    ls.N = N
    ls.z = z
    ls.name = 'test'
    return ls


def write_learning_states(path, states):
    """Write a list of LearningStates to a binary file."""
    with open(path, 'wb') as f:
        for ls in states:
            f.write(ls.to_bytes())


# ---------------------------------------------------------------------------
# make_policy_target
# ---------------------------------------------------------------------------

class TestMakePolicyTarget:
    def _counts(self, vals):
        return numpy.array(vals, dtype=numpy.uint16)

    def test_temperature_half_sums_to_one(self):
        N = self._counts([1, 2, 3, 4])
        p = make_policy_target(N, 0.5)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_temperature_one_sums_to_one(self):
        N = self._counts([1, 2, 3, 4])
        p = make_policy_target(N, 1.0)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_temperature_zero_sums_to_one(self):
        N = self._counts([1, 2, 3, 4])
        p = make_policy_target(N, 0.0)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_temperature_half_is_squared(self):
        """T=0.5 → squared counts, normalised."""
        N = self._counts([1, 2, 0, 0])
        p = make_policy_target(N, 0.5)
        # expected: [1, 4] / 5
        assert abs(p[0] - 0.2) < 1e-6
        assert abs(p[1] - 0.8) < 1e-6
        assert p[2] == 0.0

    def test_temperature_one_is_linear(self):
        """T=1.0 → linear counts, normalised."""
        N = self._counts([1, 3, 0])
        p = make_policy_target(N, 1.0)
        assert abs(p[0] - 0.25) < 1e-6
        assert abs(p[1] - 0.75) < 1e-6

    def test_temperature_zero_is_argmax(self):
        """T=0.0 → all mass on the maximum."""
        N = self._counts([1, 5, 2])
        p = make_policy_target(N, 0.0)
        assert p[1] == pytest.approx(1.0)
        assert p[0] == 0.0
        assert p[2] == 0.0

    def test_temperature_zero_ties_split(self):
        """T=0.0 → tied maxima each get equal weight."""
        N = self._counts([3, 3, 0])
        p = make_policy_target(N, 0.0)
        assert abs(p[0] - 0.5) < 1e-6
        assert abs(p[1] - 0.5) < 1e-6

    def test_output_dtype_float32(self):
        N = self._counts([1, 2, 3])
        p = make_policy_target(N, 0.5)
        assert p.dtype == numpy.float32

    def test_bad_temperature_raises(self):
        N = self._counts([1, 2, 3])
        with pytest.raises(ValueError):
            make_policy_target(N, 0.7)


# ---------------------------------------------------------------------------
# make_value_target
# ---------------------------------------------------------------------------

class TestMakeValueTarget:
    def test_loss(self):
        assert make_value_target(-1) == 0

    def test_draw(self):
        assert make_value_target(0) == 1

    def test_win(self):
        assert make_value_target(1) == 2

    def test_returns_int(self):
        assert isinstance(make_value_target(0), int)


# ---------------------------------------------------------------------------
# FileInfo
# ---------------------------------------------------------------------------

class TestFileInfo:
    def test_count_single_record(self, tmp_path):
        path = str(tmp_path / 'data.bin')
        ls = make_learning_state()
        write_learning_states(path, [ls])
        fi = FileInfo(path)
        assert fi.count == 1

    def test_count_multiple_records(self, tmp_path):
        path = str(tmp_path / 'data.bin')
        states = [make_learning_state(z) for z in [-1, 0, 1]]
        write_learning_states(path, states)
        fi = FileInfo(path)
        assert fi.count == 3

    def test_name_stored(self, tmp_path):
        path = str(tmp_path / 'data.bin')
        write_learning_states(path, [make_learning_state()])
        fi = FileInfo(path)
        assert fi.name == path

    def test_file_handle_open(self, tmp_path):
        path = str(tmp_path / 'data.bin')
        write_learning_states(path, [make_learning_state()])
        fi = FileInfo(path)
        assert not fi.f.closed


# ---------------------------------------------------------------------------
# load_selector
# ---------------------------------------------------------------------------

class TestLoadSelector:
    def test_single_file(self, tmp_path):
        import wrs
        path = str(tmp_path / 'data.bin')
        write_learning_states(path, [make_learning_state()] * 5)
        sel = wrs.WeightedRandomSelector()
        f, r = load_selector(sel, path)
        assert f == 1
        assert r == 5
        assert sel.total_weight == pytest.approx(5.0)

    def test_directory(self, tmp_path):
        import wrs
        for i in range(3):
            p = str(tmp_path / f'data{i}.bin')
            write_learning_states(p, [make_learning_state()] * 2)
        sel = wrs.WeightedRandomSelector()
        f, r = load_selector(sel, str(tmp_path))
        assert f == 3
        assert r == 6

    def test_empty_directory(self, tmp_path):
        import wrs
        sel = wrs.WeightedRandomSelector()
        f, r = load_selector(sel, str(tmp_path))
        assert f == 0
        assert r == 0


# ---------------------------------------------------------------------------
# prepare_batch
# ---------------------------------------------------------------------------

class TestPrepareBatch:
    def _batch(self, n=4, temperature=0.5, policy_epsilon=0.0):
        states = [make_learning_state(z=(i % 3) - 1, seed=i) for i in range(n)]
        return prepare_batch(states, temperature, policy_epsilon)

    def test_returns_five_tensors(self):
        result = self._batch(3)
        assert len(result) == 5

    def test_pegs_shape(self):
        pegs, *_ = self._batch(4)
        assert pegs.shape == (4, 2, SIZE, SIZE)

    def test_links_shape(self):
        _, links, *_ = self._batch(4)
        assert links.shape == (4, 8, SIZE, SIZE)

    def test_locs_shape(self):
        _, _, locs, *_ = self._batch(4)
        assert locs.shape == (4, 2, SIZE, SIZE)

    def test_p_target_shape(self):
        *_, p_target, _ = self._batch(4)
        assert p_target.shape == (4, NUM_MOVES)

    def test_v_target_shape(self):
        *_, v_target = self._batch(4)
        assert v_target.shape == (4,)

    def test_p_target_sums_to_one(self):
        *_, p_target, _ = self._batch(4)
        sums = p_target.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_v_target_valid_classes(self):
        *_, v_target = self._batch(4)
        assert v_target.min() >= 0
        assert v_target.max() <= 2

    def test_dtypes(self):
        pegs, links, locs, p_target, v_target = self._batch(2)
        assert pegs.dtype == torch.float32
        assert links.dtype == torch.float32
        assert locs.dtype == torch.float32
        assert p_target.dtype == torch.float32
        assert v_target.dtype == torch.int64

    def test_policy_epsilon_applied_at_temp_zero(self):
        """With policy_epsilon > 0 and temperature 0.0, no probability is zero."""
        states = [make_learning_state(seed=i) for i in range(2)]
        _, _, _, p_target, _ = prepare_batch(states, 0.0, policy_epsilon=0.01)
        assert (p_target > 0).all()

    def test_temperature_one(self):
        _, _, _, p_target, _ = self._batch(2, temperature=1.0)
        sums = p_target.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def make_small_model():
    return TwixNet(num_filters=8, num_blocks=1)


def make_batch_tensors(batch_size=4, device='cpu'):
    ls_list = [make_learning_state(z=(i % 3) - 1, seed=i) for i in range(batch_size)]
    return prepare_batch(ls_list, temperature=0.5, device=device)


class TestTrainer:
    def test_construction(self):
        model = make_small_model()
        t = Trainer(model, learning_rate=0.01)
        assert t is not None

    def test_get_learning_rate(self):
        model = make_small_model()
        t = Trainer(model, learning_rate=0.05)
        assert t.get_learning_rate() == pytest.approx(0.05)

    def test_set_learning_rate(self):
        model = make_small_model()
        t = Trainer(model)
        t.set_learning_rate(0.001)
        assert t.get_learning_rate() == pytest.approx(0.001)

    def test_train_step_returns_three_floats(self):
        model = make_small_model()
        t = Trainer(model)
        batch = make_batch_tensors()
        result = t.train_step(*batch)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_train_step_losses_finite(self):
        model = make_small_model()
        t = Trainer(model)
        batch = make_batch_tensors()
        total, l1, l2 = t.train_step(*batch)
        assert numpy.isfinite(total)
        assert numpy.isfinite(l1)
        assert numpy.isfinite(l2)

    def test_train_step_losses_positive(self):
        model = make_small_model()
        t = Trainer(model)
        batch = make_batch_tensors()
        total, l1, l2 = t.train_step(*batch)
        assert total > 0
        assert l1 > 0
        assert l2 > 0

    def test_train_step_increments_counter(self):
        model = make_small_model()
        t = Trainer(model)
        batch = make_batch_tensors()
        assert t.step == 0
        t.train_step(*batch)
        assert t.step == 1
        t.train_step(*batch)
        assert t.step == 2

    def test_eval_step_returns_three_floats(self):
        model = make_small_model()
        t = Trainer(model)
        batch = make_batch_tensors()
        result = t.eval_step(*batch)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_eval_step_no_gradient_update(self):
        """eval_step must not change model weights."""
        model = make_small_model()
        t = Trainer(model)
        # Snapshot weights
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        batch = make_batch_tensors()
        t.eval_step(*batch)
        for n, p in model.named_parameters():
            assert torch.equal(p, params_before[n]), \
                f"Parameter {n} changed during eval_step"

    def test_eval_step_counter_unchanged(self):
        model = make_small_model()
        t = Trainer(model)
        batch = make_batch_tensors()
        t.eval_step(*batch)
        assert t.step == 0

    def test_total_is_sum_of_components(self):
        model = make_small_model()
        t = Trainer(model)
        batch = make_batch_tensors()
        total, l1, l2 = t.eval_step(*batch)
        assert abs(total - (l1 + l2)) < 1e-5

    def test_train_step_changes_weights(self):
        """At least one parameter should change after a gradient step."""
        model = make_small_model()
        t = Trainer(model, learning_rate=0.1)
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        batch = make_batch_tensors()
        t.train_step(*batch)
        changed = any(not torch.equal(p, params_before[n])
                      for n, p in model.named_parameters())
        assert changed, "No parameter changed after train_step"

    def test_multiple_train_steps_reduce_loss(self):
        """Loss should trend downward over many batches on the same data."""
        torch.manual_seed(0)
        model = make_small_model()
        t = Trainer(model, learning_rate=0.05)
        batch = make_batch_tensors(batch_size=8)
        losses = [t.train_step(*batch)[0] for _ in range(30)]
        # Average of last 10 should be less than average of first 10
        assert numpy.mean(losses[-10:]) < numpy.mean(losses[:10]), \
            f"Loss did not decrease: {losses[:5]} ... {losses[-5:]}"


# ---------------------------------------------------------------------------
# read_from_holdout
# ---------------------------------------------------------------------------

class TestReadFromHoldout:
    def test_reads_single_file(self, tmp_path):
        path = str(tmp_path / 'hold.bin')
        states = [make_learning_state(z) for z in [-1, 0, 1, 1]]
        write_learning_states(path, states)
        result = list(read_from_holdout(path))
        assert len(result) == 4

    def test_reads_directory(self, tmp_path):
        for i in range(2):
            p = str(tmp_path / f'f{i}.bin')
            write_learning_states(p, [make_learning_state()] * 3)
        result = list(read_from_holdout(str(tmp_path)))
        assert len(result) == 6

    def test_yields_learning_states(self, tmp_path):
        path = str(tmp_path / 'data.bin')
        write_learning_states(path, [make_learning_state()])
        for ls in read_from_holdout(path):
            assert hasattr(ls, 'N')
            assert hasattr(ls, 'z')
            assert hasattr(ls, 'nips')

    def test_z_preserved(self, tmp_path):
        path = str(tmp_path / 'data.bin')
        states = [make_learning_state(z) for z in [-1, 0, 1]]
        write_learning_states(path, states)
        recovered = list(read_from_holdout(path))
        assert [ls.z for ls in recovered] == [-1, 0, 1]
