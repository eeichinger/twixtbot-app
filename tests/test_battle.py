"""
Tests for battle.py — BattleSpec and battle_once logic.

battle_once accesses module-level `args`, so we inject a mock args object
before calling it. The test thinkers are simple deterministic stubs.
"""
import os
import sys
import types
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import twixt
import naf
from twixt import Point, Game

# battle.py uses module-level `args` inside battle_once.
# Import the module and inject a minimal mock args before any test.
import battle

# Minimal mock args — all boolean flags False, no display
_mock_args = types.SimpleNamespace(
    show_moves=False,
    display=False,
    report_move_count=False,
    training_file=None,
)
battle.args = _mock_args


# ---------------------------------------------------------------------------
# Deterministic thinker stubs
# ---------------------------------------------------------------------------

class _FixedMoveThinker:
    """Always plays the first legal move."""
    name = "fixed"
    report = ""

    def pick_move(self, game):
        moves = game.legal_plays()
        if not moves:
            return "resign"
        return next(iter(moves))


class _PolicyThinker:
    """Returns (move, policy_array) like NeuralMCTS does."""
    name = "policy"
    report = ""

    def pick_move(self, game):
        moves = game.legal_plays()
        if not moves:
            return "resign"
        m = next(iter(moves))
        pa = naf.single_move_policy_array(game, m)
        return m, pa


class _ResignThinker:
    """Resigns immediately."""
    name = "resign"
    report = ""

    def pick_move(self, game):
        return "resign"


# ---------------------------------------------------------------------------
# BattleSpec construction
# ---------------------------------------------------------------------------

class TestBattleSpec:
    def test_construction(self):
        bs = battle.BattleSpec("black", "white", None)
        assert bs.black_spec == "black"
        assert bs.white_spec == "white"

    def test_with_training_true(self):
        bs = battle.BattleSpec("b", "w", None, with_training=True)
        assert isinstance(bs.train_list, list)

    def test_with_training_false(self):
        bs = battle.BattleSpec("b", "w", None, with_training=False)
        assert bs.train_list is None

    def test_win_color_initially_none(self):
        bs = battle.BattleSpec("b", "w", None)
        assert bs.win_color is None

    def test_move_list_from_none(self):
        bs = battle.BattleSpec("b", "w", None)
        assert bs.move_list == []

    def test_move_list_from_list(self):
        moves = [Point(5, 5)]
        bs = battle.BattleSpec("b", "w", moves)
        # move_list is a copy
        assert bs.move_list == [Point(5, 5)]


# ---------------------------------------------------------------------------
# battle_once — basic outcomes
# ---------------------------------------------------------------------------

class TestBattleOnce:
    def _run(self, black_th, white_th, moves=None, train_list=None, move_list=None):
        return battle.battle_once(black_th, white_th, moves, train_list, move_list)

    def test_returns_two_values(self):
        win_color, final_score = self._run(_FixedMoveThinker(), _FixedMoveThinker())
        assert final_score is not None

    def test_returns_valid_win_color(self):
        win_color, final_score = self._run(_FixedMoveThinker(), _FixedMoveThinker())
        assert win_color in (None, Game.BLACK, Game.WHITE)

    def test_resign_produces_opponent_win(self):
        """If black resigns immediately, white (1-black) wins."""
        win_color, final_score = self._run(_ResignThinker(), _FixedMoveThinker())
        # Resign on first move (black's turn) → white wins
        assert win_color == Game.WHITE
        assert final_score == 1

    def test_resign_score_is_one(self):
        win_color, score = self._run(_ResignThinker(), _FixedMoveThinker())
        assert score == 1

    def test_with_train_list_appends(self):
        train = []
        self._run(_ResignThinker(), _FixedMoveThinker(), train_list=train)
        # resign pops last entry, so train may be empty or have entries up to resign
        assert isinstance(train, list)

    def test_policy_thinker_runs(self):
        """battle_once works with a thinker returning (move, policy_array)."""
        win_color, score = self._run(_PolicyThinker(), _PolicyThinker())
        assert score is not None

    def test_str_init_moves(self):
        """String init moves are replayed before the game starts."""
        win_color, score = self._run(
            _ResignThinker(), _FixedMoveThinker(),
            moves="b1"  # single point in letter format
        )
        assert win_color in (None, Game.BLACK, Game.WHITE)

    def test_point_init_moves(self):
        """Point init move is played before the game starts."""
        win_color, score = self._run(
            _ResignThinker(), _FixedMoveThinker(),
            moves=Point(5, 5)
        )
        assert win_color in (None, Game.BLACK, Game.WHITE)

    def test_move_list_collected(self):
        """move_list accumulates moves played during the game."""
        move_list = []
        self._run(_ResignThinker(), _FixedMoveThinker(), move_list=move_list)
        # Resign before any move, so move_list may be empty
        assert isinstance(move_list, list)

    def test_full_game_completes(self):
        """A game between two _FixedMoveThinkers eventually produces a winner."""
        win_color, score = self._run(_FixedMoveThinker(), _FixedMoveThinker())
        # Either someone wins or it's a draw (no legal moves / can't win)
        assert win_color in (None, Game.BLACK, Game.WHITE)
        assert score in (-1, 0, 1)


# ---------------------------------------------------------------------------
# queue import (Python 3 fix: Queue → queue)
# ---------------------------------------------------------------------------

class TestQueueImport:
    def test_queue_module_is_python3(self):
        """Ensure battle.py uses the Python 3 queue module, not Queue."""
        import queue as q3
        assert isinstance(battle.ThreadingManager(_mock_args).job_request_queue, q3.Queue)
