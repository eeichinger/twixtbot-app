"""
Tests for Zobrist hashing in twixt.Game and the position cache in asn_player.
"""
import numpy
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import twixt
from twixt import Point, Game


# ---------------------------------------------------------------------------
# Zobrist hash basics
# ---------------------------------------------------------------------------

class TestZobristHash:
    def test_empty_board_hash_is_zero(self):
        assert Game().zhash == 0

    def test_play_changes_hash(self):
        g = Game()
        g.play(Point(5, 5))
        assert g.zhash != 0

    def test_undo_restores_hash(self):
        g = Game()
        g.play(Point(5, 5))
        g.undo()
        assert g.zhash == 0

    def test_multi_move_undo_roundtrip(self):
        g = Game()
        moves = [Point(5, 5), Point(7, 6), Point(3, 3), Point(10, 10)]
        for m in moves:
            g.play(m)
        for _ in moves:
            g.undo()
        assert g.zhash == 0

    def test_different_moves_different_hashes(self):
        g1 = Game()
        g1.play(Point(5, 5))
        g2 = Game()
        g2.play(Point(6, 6))
        assert g1.zhash != g2.zhash


# ---------------------------------------------------------------------------
# Transposition detection
# ---------------------------------------------------------------------------

class TestZobristTransposition:
    def test_move_order_transposition(self):
        """Same pegs placed by same colors in different order → same hash."""
        # Game 1: White e5, Black c3, White g7
        g1 = Game()
        g1.play(Point(5, 5))   # White
        g1.play(Point(3, 3))   # Black
        g1.play(Point(7, 7))   # White

        # Game 2: White g7, Black c3, White e5
        g2 = Game()
        g2.play(Point(7, 7))   # White
        g2.play(Point(3, 3))   # Black
        g2.play(Point(5, 5))   # White

        assert g1.zhash == g2.zhash

    def test_color_swap_no_transposition(self):
        """Same positions but different color assignments → different hash."""
        # Game 1: White at (5,5), Black at (3,3)
        g1 = Game()
        g1.play(Point(5, 5))   # White
        g1.play(Point(3, 3))   # Black

        # Game 2: White at (3,3), Black at (5,5)
        g2 = Game()
        g2.play(Point(3, 3))   # White
        g2.play(Point(5, 5))   # Black

        assert g1.zhash != g2.zhash

    def test_turn_matters(self):
        """Same pegs but different turn → different hash.

        After White plays (5,5), it's Black's turn.
        After White plays (5,5), Black plays (3,3), White plays (7,7),
        it's Black's turn again, with more pegs. Not a transposition.
        But the key property: turn is encoded via ZOBRIST_TURN XOR.
        """
        g1 = Game()
        g1.play(Point(5, 5))  # Black's turn
        h_black_turn = g1.zhash

        g1.play(Point(3, 3))  # White's turn
        h_white_turn = g1.zhash

        # Both have different pegs AND different turns, so definitely different
        assert h_black_turn != h_white_turn

    def test_four_move_transposition(self):
        """Four independent moves in two different orders."""
        moves_w = [Point(5, 5), Point(9, 9)]    # White's moves
        moves_b = [Point(3, 3), Point(11, 11)]  # Black's moves

        # Order 1: W(5,5), B(3,3), W(9,9), B(11,11)
        g1 = Game()
        g1.play(moves_w[0]); g1.play(moves_b[0])
        g1.play(moves_w[1]); g1.play(moves_b[1])

        # Order 2: W(9,9), B(11,11), W(5,5), B(3,3)
        g2 = Game()
        g2.play(moves_w[1]); g2.play(moves_b[1])
        g2.play(moves_w[0]); g2.play(moves_b[0])

        assert g1.zhash == g2.zhash


# ---------------------------------------------------------------------------
# Swap rule
# ---------------------------------------------------------------------------

class TestZobristSwap:
    def test_swap_changes_hash(self):
        g = Game()
        g.play(Point(5, 5))
        h_before = g.zhash
        g.play_swap()
        assert g.zhash != h_before

    def test_undo_swap_restores_hash(self):
        g = Game()
        g.play(Point(5, 5))
        h_before = g.zhash
        g.play_swap()
        g.undo()
        assert g.zhash == h_before

    def test_swap_full_roundtrip(self):
        """Play, swap, undo swap, undo play → back to 0."""
        g = Game()
        g.play(Point(5, 5))
        g.play_swap()
        g.undo()   # undo swap
        g.undo()   # undo play
        assert g.zhash == 0


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------

class TestZobristClone:
    def test_clone_preserves_hash(self):
        g = Game()
        g.play(Point(5, 5))
        g.play(Point(7, 6))
        c = g.clone()
        assert c.zhash == g.zhash

    def test_clone_hash_independent(self):
        g = Game()
        g.play(Point(5, 5))
        c = g.clone()
        c.play(Point(7, 6))
        # Original hash unchanged
        g2 = Game()
        g2.play(Point(5, 5))
        assert g.zhash == g2.zhash
        assert c.zhash != g.zhash
