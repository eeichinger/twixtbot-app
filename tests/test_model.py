"""
Tests for model.py — TwixNet PyTorch implementation.

Architecture mirrors mkbig.py (TF1 original) with these intentional changes:
  - Activation: GELU instead of abs()  (modernisation)
  - Policy head reshape: 528 (correct) instead of 529 (original off-by-one bug)
  - Value head: always 3-class (Loss/Draw/Win) output

All tests run on CPU; no GPU required.
"""
import math
import numpy
import pytest
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import twixt
import naf
from twixt import Point, Game
from naf import LearningState
import model as mdl
from model import TwixNet, ResidualBlock, make_sap

SIZE = Game.SIZE
NUM_MOVES = LearningState.NUM_MOVES  # 528
F = 40   # default num_filters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(batch_size=2, device='cpu'):
    """Return random (pegs, links, locs) tensors in NCHW float32."""
    pegs  = torch.randn(batch_size, 2, SIZE, SIZE, device=device)
    links = torch.randn(batch_size, 8, SIZE, SIZE, device=device)
    locs  = torch.randn(batch_size, 2, SIZE, SIZE, device=device)
    return pegs, links, locs


def make_model(**kwargs):
    return TwixNet(**kwargs).eval()


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class TestResidualBlock:
    def test_output_shape_preserved(self):
        block = ResidualBlock(F)
        x = torch.randn(2, F, SIZE, SIZE)
        out = block(x)
        assert out.shape == (2, F, SIZE, SIZE)

    def test_output_finite(self):
        block = ResidualBlock(F)
        x = torch.randn(2, F, SIZE, SIZE)
        out = block(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        block = ResidualBlock(F).train()
        x = torch.randn(2, F, SIZE, SIZE, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# TwixNet output shapes
# ---------------------------------------------------------------------------

class TestTwixNetShapes:
    def test_policy_shape(self):
        m = make_model()
        pegs, links, locs = make_batch(batch_size=3)
        policy, value = m(pegs, links, locs)
        assert policy.shape == (3, NUM_MOVES), \
            f"Expected (3, {NUM_MOVES}), got {policy.shape}"

    def test_value_shape(self):
        m = make_model()
        pegs, links, locs = make_batch(batch_size=3)
        policy, value = m(pegs, links, locs)
        assert value.shape == (3, 3), \
            f"Expected (3, 3) for 3-class value, got {value.shape}"

    def test_batch_size_1(self):
        m = make_model()
        pegs, links, locs = make_batch(batch_size=1)
        policy, value = m(pegs, links, locs)
        assert policy.shape == (1, NUM_MOVES)
        assert value.shape == (1, 3)

    def test_batch_size_8(self):
        m = make_model()
        pegs, links, locs = make_batch(batch_size=8)
        policy, value = m(pegs, links, locs)
        assert policy.shape == (8, NUM_MOVES)
        assert value.shape == (8, 3)

    def test_policy_num_moves_is_528(self):
        """Policy output must have exactly 528 moves — not 529 (the original bug)."""
        m = make_model()
        pegs, links, locs = make_batch()
        policy, _ = m(pegs, links, locs)
        assert policy.shape[-1] == 528


# ---------------------------------------------------------------------------
# TwixNet output values
# ---------------------------------------------------------------------------

class TestTwixNetOutputs:
    def test_output_dtype_float32(self):
        m = make_model()
        pegs, links, locs = make_batch()
        policy, value = m(pegs, links, locs)
        assert policy.dtype == torch.float32
        assert value.dtype == torch.float32

    def test_outputs_finite(self):
        m = make_model()
        pegs, links, locs = make_batch()
        policy, value = m(pegs, links, locs)
        assert torch.isfinite(policy).all(), "Policy output contains NaN/Inf"
        assert torch.isfinite(value).all(), "Value output contains NaN/Inf"

    def test_policy_not_all_zero(self):
        """With random weights and random input, policy should not be all zeros."""
        torch.manual_seed(0)
        m = make_model()
        pegs, links, locs = make_batch()
        policy, _ = m(pegs, links, locs)
        assert policy.abs().sum() > 0

    def test_deterministic_eval_mode(self):
        """Same input → same output in eval mode (no dropout/stochastic BN)."""
        m = make_model()
        pegs, links, locs = make_batch(batch_size=1)
        with torch.no_grad():
            p1, v1 = m(pegs, links, locs)
            p2, v2 = m(pegs, links, locs)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)


# ---------------------------------------------------------------------------
# Gradient / training
# ---------------------------------------------------------------------------

class TestGradients:
    def test_backward_policy_loss(self):
        """Policy loss backward gives gradients to shared trunk and policy head.

        Value-head parameters have no gradient from a pure policy loss — they are
        in a separate branch. The combined loss test verifies all params are reached.
        """
        m = TwixNet().train()
        pegs, links, locs = make_batch(batch_size=4)
        policy, value = m(pegs, links, locs)
        target = torch.full((4, NUM_MOVES), 1.0 / NUM_MOVES)
        loss = torch.nn.functional.cross_entropy(policy, target)
        loss.backward()
        # Check shared trunk (primary + blocks) and policy head got gradients
        policy_params = {n for n, _ in m.named_parameters()
                         if any(n.startswith(p) for p in
                                ('primary_', 'blocks.', 'policy_'))}
        for name, param in m.named_parameters():
            if name in policy_params:
                assert param.grad is not None, f"No grad for shared/policy param {name}"

    def test_backward_value_loss(self):
        """Cross-entropy value loss backward should not raise."""
        m = TwixNet().train()
        pegs, links, locs = make_batch(batch_size=4)
        _, value = m(pegs, links, locs)
        target = torch.zeros(4, dtype=torch.long)  # all "loss" class
        loss = torch.nn.functional.cross_entropy(value, target)
        loss.backward()

    def test_backward_combined_loss(self):
        """Combined policy+value loss backward: all parameters get gradients."""
        m = TwixNet().train()
        pegs, links, locs = make_batch(batch_size=4)
        policy, value = m(pegs, links, locs)
        p_target = torch.full((4, NUM_MOVES), 1.0 / NUM_MOVES)
        v_target = torch.zeros(4, dtype=torch.long)
        loss = (torch.nn.functional.cross_entropy(policy, p_target)
                + torch.nn.functional.cross_entropy(value, v_target))
        loss.backward()
        params_with_grad = sum(1 for p in m.parameters() if p.grad is not None)
        total_params = sum(1 for _ in m.parameters())
        assert params_with_grad == total_params, \
            f"Only {params_with_grad}/{total_params} parameters got gradients"


# ---------------------------------------------------------------------------
# Hyperparameter flexibility
# ---------------------------------------------------------------------------

class TestHyperparams:
    def test_fewer_filters(self):
        m = TwixNet(num_filters=16, num_blocks=2).eval()
        pegs, links, locs = make_batch()
        policy, value = m(pegs, links, locs)
        assert policy.shape == (2, NUM_MOVES)
        assert value.shape == (2, 3)

    def test_more_blocks(self):
        m = TwixNet(num_filters=16, num_blocks=4).eval()
        pegs, links, locs = make_batch()
        policy, value = m(pegs, links, locs)
        assert policy.shape[-1] == NUM_MOVES

    def test_abs_activation(self):
        """Model should work with abs activation (original behaviour)."""
        m = TwixNet(activation='abs').eval()
        pegs, links, locs = make_batch()
        policy, value = m(pegs, links, locs)
        assert torch.isfinite(policy).all()

    def test_silu_activation(self):
        m = TwixNet(activation='silu').eval()
        pegs, links, locs = make_batch()
        policy, value = m(pegs, links, locs)
        assert torch.isfinite(policy).all()

    def test_one_value_reduction(self):
        m = TwixNet(num_filters=16, num_blocks=2, value_reductions=1).eval()
        pegs, links, locs = make_batch()
        policy, value = m(pegs, links, locs)
        assert value.shape == (2, 3)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_state_dict_roundtrip(self):
        """Save and reload state_dict — outputs must be identical."""
        m = make_model()
        pegs, links, locs = make_batch(batch_size=1)
        with torch.no_grad():
            p_before, v_before = m(pegs, links, locs)

        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)

        m2 = make_model()
        m2.load_state_dict(torch.load(buf, weights_only=True))
        m2.eval()
        with torch.no_grad():
            p_after, v_after = m2(pegs, links, locs)

        assert torch.allclose(p_before, p_after, atol=1e-6)
        assert torch.allclose(v_before, v_after, atol=1e-6)

    def test_save_load_preserves_architecture(self):
        """Loading into a different-but-compatible model works."""
        m1 = TwixNet(num_filters=16, num_blocks=2)
        m2 = TwixNet(num_filters=16, num_blocks=2)
        buf = io.BytesIO()
        torch.save(m1.state_dict(), buf)
        buf.seek(0)
        m2.load_state_dict(torch.load(buf, weights_only=True))


# ---------------------------------------------------------------------------
# sap integration
# ---------------------------------------------------------------------------

class TestSap:
    def test_sap_returns_scalar_and_array(self):
        m = make_model()
        sap = make_sap(m)
        g = twixt.Game()
        score, policy = sap(g)
        assert isinstance(score, float), f"score should be float, got {type(score)}"
        assert isinstance(policy, numpy.ndarray)
        assert policy.shape == (NUM_MOVES,)

    def test_sap_score_in_range(self):
        """Score from three_to_one should be in (-1, 1)."""
        m = make_model()
        sap = make_sap(m)
        g = twixt.Game()
        score, _ = sap(g)
        assert -1.0 <= score <= 1.0

    def test_sap_policy_finite(self):
        m = make_model()
        sap = make_sap(m)
        g = twixt.Game()
        _, policy = sap(g)
        assert numpy.isfinite(policy).all()

    def test_sap_game_state_unchanged(self):
        """sap must not modify the game state."""
        m = make_model()
        sap = make_sap(m)
        g = twixt.Game()
        # Take a snapshot
        pegs_before = [p.copy() for p in g.pegs]
        turn_before = g.turn
        sap(g)
        assert g.turn == turn_before
        for i in range(2):
            assert numpy.array_equal(g.pegs[i], pegs_before[i])

    def test_sap_works_after_moves(self):
        """sap works on non-empty game positions."""
        m = make_model()
        sap = make_sap(m)
        g = twixt.Game()
        g.play(Point(5, 5))
        g.play(Point(10, 10))
        score, policy = sap(g)
        assert isinstance(score, float)
        assert policy.shape == (NUM_MOVES,)

    def test_sap_usable_in_mcts(self):
        """make_sap output is compatible with NeuralMCTS."""
        import nnmcts
        m = make_model()
        sap = make_sap(m)
        g = twixt.Game()
        nm = nnmcts.NeuralMCTS(sap, add_noise=0)
        result = nm.mcts(g, trials=3)
        # Should return an ndarray (no forced win in 3 trials on empty board)
        assert isinstance(result, numpy.ndarray)
        assert result.shape == (NUM_MOVES,)

    def test_sap_black_to_move(self):
        """sap works correctly when BLACK is to move (transposed perspective)."""
        m = make_model()
        sap = make_sap(m)
        g = twixt.Game()
        g.play(Point(5, 5))   # WHITE plays; now BLACK to move
        score, policy = sap(g)
        assert isinstance(score, float)
        assert policy.shape == (NUM_MOVES,)

    def test_sap_device_param(self):
        """make_sap accepts a device parameter without error."""
        m = make_model()
        sap = make_sap(m, device='cpu')
        g = twixt.Game()
        score, policy = sap(g)
        assert policy.shape == (NUM_MOVES,)
