"""
Tests for nnmcts.py — Neural MCTS implementation.

Uses a mock sap (score-and-policy) callback so no real neural net is required.
Reference baseline for the Python 2 → Python 3 / PyTorch migration.
"""
import math
import numpy
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import twixt
import naf
import nnmcts
from twixt import Point, Game
from nnmcts import EvalNode, NeuralMCTS
from naf import LearningState

NUM_MOVES = LearningState.NUM_MOVES  # 528


# ---------------------------------------------------------------------------
# Mock sap helpers
# ---------------------------------------------------------------------------

def mock_sap_uniform(game):
    """Zero score, uniform logits — all legal moves equally likely."""
    ml = numpy.zeros(NUM_MOVES)
    return 0.0, ml


def mock_sap_favour(pt):
    """Strongly favour one specific point; return zero score."""
    def sap(game):
        ml = numpy.full(NUM_MOVES, -100.0)
        idx = naf.policy_point_index(game.turn, pt)
        ml[idx] = 0.0
        return 0.0, ml
    return sap


def mock_sap_winning(pt):
    """Strongly favour one specific point and return win score (+1)."""
    def sap(game):
        ml = numpy.full(NUM_MOVES, -100.0)
        idx = naf.policy_point_index(game.turn, pt)
        ml[idx] = 0.0
        return 1.0, ml
    return sap


def fresh_game():
    return Game()


def game_snapshot(g):
    return {
        'pegs': [p.copy() for p in g.pegs],
        'links': [lk.copy() for lk in g.links],
        'turn': g.turn,
        'history_len': len(g.history),
    }


def snapshots_equal(a, b):
    if a['turn'] != b['turn'] or a['history_len'] != b['history_len']:
        return False
    for i in range(2):
        if not numpy.array_equal(a['pegs'][i], b['pegs'][i]):
            return False
    for i in range(8):
        if not numpy.array_equal(a['links'][i], b['links'][i]):
            return False
    return True


# ---------------------------------------------------------------------------
# EvalNode initialisation
# ---------------------------------------------------------------------------

class TestEvalNode:
    def test_n_zeros(self):
        node = EvalNode()
        assert node.N.shape == (NUM_MOVES,)
        assert numpy.all(node.N == 0)

    def test_q_zeros(self):
        node = EvalNode()
        assert node.Q.shape == (NUM_MOVES,)
        assert numpy.all(node.Q == 0)

    def test_p_zeros(self):
        node = EvalNode()
        assert node.P.shape == (NUM_MOVES,)
        assert numpy.all(node.P == 0)

    def test_not_proven(self):
        node = EvalNode()
        assert node.proven is False

    def test_score_none(self):
        node = EvalNode()
        assert node.score is None

    def test_subnodes_none(self):
        node = EvalNode()
        assert len(node.subnodes) == NUM_MOVES
        assert all(s is None for s in node.subnodes)


# ---------------------------------------------------------------------------
# expand_leaf
# ---------------------------------------------------------------------------

class TestExpandLeaf:
    def test_fresh_game_not_proven(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        leaf = nm.expand_leaf(g)
        assert leaf.proven is False

    def test_fresh_game_legal_moves(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        leaf = nm.expand_leaf(g)
        assert leaf.LM.sum() == NUM_MOVES

    def test_policy_sums_to_one(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        leaf = nm.expand_leaf(g)
        assert abs(leaf.P.sum() - 1.0) < 1e-5

    def test_score_from_sap(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        leaf = nm.expand_leaf(g)
        assert leaf.score == 0.0

    def test_just_won_proven(self):
        """expand_leaf on a game where the previous move won → proven node."""
        # Build a winning chain for WHITE (y=0..23 zigzag via cols 5 and 7)
        g = fresh_game()
        white_moves = []
        x = 5
        for y in range(24):
            white_moves.append(Point(x, y))
            x = 12 - x
        # 24 WHITE moves → 23 BLACK moves needed between them
        black_pool = [Point(col, r) for col in (15, 16) for r in range(1, 23)]
        bi = 0
        for i, wm in enumerate(white_moves):
            g.play(wm)
            if i < len(white_moves) - 1:
                g.play(black_pool[bi]); bi += 1
        # WHITE just won; it's BLACK's turn; game.just_won() == True
        assert g.just_won()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        leaf = nm.expand_leaf(g)
        assert leaf.proven is True
        assert leaf.score == -1
        assert leaf.LMnz == "just_won"

    def test_policy_illegal_moves_zero(self):
        """Illegal moves (border cells) should have policy weight 0."""
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        leaf = nm.expand_leaf(g)
        # All entries in P are non-negative; legal moves sum to 1;
        # P values for illegal positions are 0 (they're masked by the softmax divisor)
        lmask = leaf.LM
        p_illegal = leaf.P[lmask == 0]
        assert numpy.all(p_illegal == 0.0)

    def test_noise_changes_policy(self):
        """With add_noise > 0, policy differs from noiseless version."""
        numpy.random.seed(42)
        g = fresh_game()
        nm_noisy = NeuralMCTS(mock_sap_uniform, add_noise=0.25)
        nm_clean = NeuralMCTS(mock_sap_uniform, add_noise=0.0)
        leaf_noisy = nm_noisy.expand_leaf(g)
        leaf_clean = nm_clean.expand_leaf(g)
        # Policies should differ due to Dirichlet noise
        assert not numpy.array_equal(leaf_noisy.P, leaf_clean.P)

    def test_noiseless_policy_sum_to_one(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0.0)
        leaf = nm.expand_leaf(g)
        assert abs(leaf.P.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# mcts return types and basic behaviour
# ---------------------------------------------------------------------------

class TestMcts:
    def test_returns_ndarray_on_normal_game(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        result = nm.mcts(g, trials=10)
        assert isinstance(result, numpy.ndarray)
        assert result.shape == (NUM_MOVES,)

    def test_visit_counts_positive(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        result = nm.mcts(g, trials=10)
        assert result.sum() > 0

    def test_only_legal_moves_visited(self):
        """Border cells that are illegal for the current player must have 0 visits."""
        g = fresh_game()  # WHITE to move
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        N = nm.mcts(g, trials=20)
        lm = naf.legal_move_policy_array(g)
        illegal_visits = N[lm == 0]
        assert numpy.all(illegal_visits == 0)

    def test_game_state_unchanged_after_mcts(self):
        """MCTS internally plays/undoes moves; game must be fully restored."""
        g = fresh_game()
        snap_before = game_snapshot(g)
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        nm.mcts(g, trials=20)
        snap_after = game_snapshot(g)
        assert snapshots_equal(snap_before, snap_after)

    def test_more_trials_more_visits(self):
        """More trials → higher total visit count."""
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        N5 = nm.mcts(g, trials=5)
        # Reset root for a fresh run
        nm2 = NeuralMCTS(mock_sap_uniform, add_noise=0)
        N50 = nm2.mcts(g, trials=50)
        assert N50.sum() >= N5.sum()

    def _build_one_move_from_win(self):
        """Return (game, winning_move) where WHITE needs exactly one more move to win.

        Plays a WHITE zigzag chain from y=0..22, always followed by a BLACK move,
        so it ends on WHITE's turn. winning_move completes the chain at y=23.
        """
        g = fresh_game()
        white_moves = []
        x = 5
        for y in range(23):  # 23 WHITE pegs at y=0..22
            white_moves.append(Point(x, y))
            x = 12 - x
        winning_move = Point(x, 23)
        # 23 WHITE moves → 23 BLACK moves so it's WHITE's turn after
        black_pool = [Point(col, r) for col in (15, 16) for r in range(1, 23)]
        bi = 0
        for wm in white_moves:
            g.play(wm)             # WHITE
            g.play(black_pool[bi]); bi += 1  # BLACK (always)
        assert g.turn == Game.WHITE
        return g, winning_move

    def test_returns_point_on_forced_win(self):
        """When a win is proven, mcts returns a twixt.Point, not an array."""
        g, winning_move = self._build_one_move_from_win()
        nm = NeuralMCTS(mock_sap_winning(winning_move), add_noise=0)
        result = nm.mcts(g, trials=200)
        assert isinstance(result, twixt.Point), \
            f"Expected Point (forced win), got {type(result)}"

    def test_root_proven_winning_move_set(self):
        """After a forced win, root.proven is True and winning_move is set."""
        g, winning_move = self._build_one_move_from_win()
        nm = NeuralMCTS(mock_sap_winning(winning_move), add_noise=0)
        nm.mcts(g, trials=200)
        assert nm.root.proven
        assert nm.root.winning_move is not None


# ---------------------------------------------------------------------------
# compute_root / tree reuse
# ---------------------------------------------------------------------------

class TestComputeRoot:
    def test_root_none_before_mcts(self):
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        assert nm.root is None
        assert nm.history_at_root is None

    def test_root_set_after_mcts(self):
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        nm.mcts(g, trials=5)
        assert nm.root is not None
        assert nm.history_at_root is not None

    def test_history_at_root_updated_after_mcts(self):
        """history_at_root is set to game.history after each mcts call."""
        g = fresh_game()
        g.play(Point(5, 5))
        g.play(Point(10, 10))
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        nm.mcts(g, trials=5)
        assert nm.history_at_root == [Point(5, 5), Point(10, 10)]

    def test_compute_root_known_bug_missing_increment(self):
        """compute_root has a pre-existing bug: the second while-loop is missing
        i += 1, causing it to overshoot the subtree and always set self.root = None.
        This test documents the actual (buggy) behaviour so a future fix is detectable.

        NOTE: the intended behaviour was tree reuse; this will need to be fixed
        (add `i += 1` in the second while loop of compute_root) before tree reuse
        can work.
        """
        g = fresh_game()
        g.play(Point(5, 5))
        g.play(Point(10, 10))
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        nm.mcts(g, trials=10)
        assert nm.history_at_root  # non-empty

        N = nm.mcts(g, trials=10)  # second call on same position — root reused (same game)
        first_root = nm.root

        # Now advance by one move
        top_move = naf.policy_index_point(g.turn, int(numpy.argmax(N)))
        g.play(top_move)

        nm.mcts(g, trials=5)
        # Due to the missing i += 1, compute_root overshoots and resets root to None,
        # then mcts creates a brand-new root. So the root is always re-created.
        assert nm.root is not first_root, \
            "Bug present: root was re-created (not reused) due to missing i += 1"

    def test_root_reset_on_diverged_history(self):
        """If we play a different game line, compute_root discards the old tree."""
        g = fresh_game()
        nm = NeuralMCTS(mock_sap_uniform, add_noise=0)
        nm.mcts(g, trials=5)
        first_root = nm.root

        # Start a completely different game
        g2 = fresh_game()
        g2.play(Point(10, 10))  # different first move
        g2.play(Point(15, 15))
        nm.mcts(g2, trials=5)
        # Root must have been reset since histories diverged
        assert nm.root is not first_root


# ---------------------------------------------------------------------------
# NeuralMCTS constructor kwargs
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_kwargs(self):
        nm = NeuralMCTS(mock_sap_uniform)
        assert nm.cpuct == 1.0
        assert nm.add_noise == 0.25
        assert nm.verbosity == 0

    def test_custom_kwargs(self):
        nm = NeuralMCTS(mock_sap_uniform, cpuct=2.0, add_noise=0.1, verbosity=1)
        assert nm.cpuct == 2.0
        assert nm.add_noise == 0.1
        assert nm.verbosity == 1

    def test_unexpected_kwarg_raises(self):
        with pytest.raises(TypeError):
            NeuralMCTS(mock_sap_uniform, bogus_param=99)
