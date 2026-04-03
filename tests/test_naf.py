"""
Tests for naf.py — board encoding for the neural network.

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
from twixt import Point, Game
from naf import (
    NetInputs,
    LearningState,
    policy_index_point,
    policy_point_index,
    legal_move_policy_array,
    single_move_policy_array,
    rotate_policy_array,
    hflip_policy_array,
    vflip_policy_array,
    three_to_one,
    one_to_three,
    location_inputs,
    NUM_ROTATIONS,
)

SIZE = Game.SIZE
NUM_MOVES = LearningState.NUM_MOVES  # 528


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_game():
    return Game()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_num_moves(self):
        assert NUM_MOVES == SIZE * (SIZE - 2)
        assert NUM_MOVES == 528

    def test_compact_size(self):
        assert NetInputs.COMPACT_SIZE == NetInputs.FRONT_BYTES + 10 * SIZE * SIZE // 8

    def test_naf_dims(self):
        assert NetInputs.NAF_DIMS == (SIZE, SIZE, 11)


# ---------------------------------------------------------------------------
# policy_index_point / policy_point_index round-trips
# ---------------------------------------------------------------------------

class TestPolicyIndex:
    def test_white_round_trip_all(self):
        """All 528 WHITE indices survive policy_index_point → policy_point_index."""
        for idx in range(NUM_MOVES):
            pt = policy_index_point(Game.WHITE, idx)
            back = policy_point_index(Game.WHITE, pt)
            assert back == idx, f"WHITE index {idx} → {pt} → {back}"

    def test_black_round_trip_all(self):
        """All 528 BLACK indices survive the round-trip."""
        for idx in range(NUM_MOVES):
            pt = policy_index_point(Game.BLACK, idx)
            back = policy_point_index(Game.BLACK, pt)
            assert back == idx, f"BLACK index {idx} → {pt} → {back}"

    def test_white_first_index(self):
        # WHITE index 0: major=0 → x=1, minor=0 → y=0
        assert policy_index_point(Game.WHITE, 0) == Point(1, 0)

    def test_black_first_index(self):
        # BLACK index 0: major=0 → y=1, minor=0 → x=0
        assert policy_index_point(Game.BLACK, 0) == Point(0, 1)

    def test_white_inbounds_all(self):
        """All WHITE policy points must be in-bounds for WHITE."""
        for idx in range(NUM_MOVES):
            pt = policy_index_point(Game.WHITE, idx)
            assert Game.inbounds_for_player(pt, Game.WHITE), \
                f"WHITE index {idx} → {pt} not in bounds"

    def test_black_inbounds_all(self):
        """All BLACK policy points must be in-bounds for BLACK."""
        for idx in range(NUM_MOVES):
            pt = policy_index_point(Game.BLACK, idx)
            assert Game.inbounds_for_player(pt, Game.BLACK), \
                f"BLACK index {idx} → {pt} not in bounds"

    def test_accepts_game_object(self):
        g = fresh_game()
        pt = policy_index_point(g, 0)
        assert pt == Point(1, 0)  # WHITE to move

    def test_bad_type_raises(self):
        with pytest.raises(ValueError):
            policy_index_point("bad", 0)


# ---------------------------------------------------------------------------
# legal_move_policy_array
# ---------------------------------------------------------------------------

class TestLegalMovePolicyArray:
    def test_empty_board_all_legal(self):
        g = fresh_game()
        arr = legal_move_policy_array(g)
        assert arr.shape == (NUM_MOVES,)
        assert numpy.all(arr == 1)

    def test_played_cell_becomes_illegal(self):
        g = fresh_game()
        pt = Point(5, 5)
        g.play(pt)
        # Now it's BLACK's turn
        arr = legal_move_policy_array(g)
        # pt should be 0 in BLACK's policy array
        idx = policy_point_index(Game.BLACK, pt)
        assert arr[idx] == 0

    def test_mask_matches_open_pegs_white(self):
        """For WHITE, mask=1 iff point is in open_pegs[WHITE]."""
        g = fresh_game()
        g.play(Point(5, 5))
        g.undo()
        arr = legal_move_policy_array(g)  # WHITE to move
        for idx in range(NUM_MOVES):
            pt = policy_index_point(Game.WHITE, idx)
            expected = 1 if pt in g.open_pegs[Game.WHITE] else 0
            assert arr[idx] == expected, f"idx={idx}, pt={pt}"

    def test_mask_matches_open_pegs_black(self):
        """For BLACK, mask=1 iff point is in open_pegs[BLACK]."""
        g = fresh_game()
        g.play(Point(5, 5))  # WHITE; now BLACK's turn
        arr = legal_move_policy_array(g)
        for idx in range(NUM_MOVES):
            pt = policy_index_point(Game.BLACK, idx)
            expected = 1 if pt in g.open_pegs[Game.BLACK] else 0
            assert arr[idx] == expected, f"idx={idx}, pt={pt}"


# ---------------------------------------------------------------------------
# NetInputs shape and contents
# ---------------------------------------------------------------------------

class TestNetInputs:
    def test_shape(self):
        g = fresh_game()
        ni = NetInputs(g)
        assert ni.naf.shape == (SIZE, SIZE, 11)

    def test_empty_board_all_zero_except_recents(self):
        g = fresh_game()
        ni = NetInputs(g)
        # planes 0-9 should all be zero on empty board
        assert numpy.all(ni.naf[:, :, :10] == 0)
        # plane 10 (recents): also zero on empty board with no history
        assert numpy.all(ni.naf[:, :, 10] == 0)

    def test_white_peg_plane_white_to_move(self):
        """After WHITE plays and it's BLACK's turn, WHITE peg appears in plane 8 (opponent)."""
        g = fresh_game()
        pt = Point(5, 5)
        g.play(pt)   # WHITE plays → BLACK to move
        # plane 8 = OPPONENT's pegs; WHITE is opponent when BLACK moves
        ni = NetInputs(g)  # BLACK to move
        assert ni.naf[pt.x, pt.y, 8] == 1   # WHITE peg in plane 8 (opponent)
        # BLACK has no pegs yet → plane 9 (current player) all zero
        assert numpy.count_nonzero(ni.naf[:, :, 9]) == 0

    def test_peg_planes_white_perspective(self):
        """When WHITE is to move: plane 8 = BLACK pegs (opponent), plane 9 = WHITE pegs (current)."""
        g = fresh_game()
        wp = Point(5, 5)
        g.play(wp)   # WHITE plays
        bp = Point(10, 10)
        g.play(bp)   # BLACK plays → now WHITE to move
        ni = NetInputs(g)
        assert ni.naf[wp.x, wp.y, 9] == 1   # WHITE peg (current player) in plane 9
        assert ni.naf[bp.x, bp.y, 8] == 1   # BLACK peg (opponent) in plane 8

    def test_peg_planes_black_perspective(self):
        """When BLACK is to move: board is transposed; plane 8 = opponent (WHITE.T)."""
        g = fresh_game()
        # Use asymmetric point so transpose is detectable
        wp = Point(3, 7)
        g.play(wp)   # WHITE → BLACK to move
        ni = NetInputs(g)
        # plane 8 = opponent (WHITE) pegs, transposed: Point(3,7) appears at [7,3]
        assert ni.naf[wp.y, wp.x, 8] == 1   # transposed: x and y swapped
        # No BLACK pegs yet → plane 9 all zero
        assert numpy.count_nonzero(ni.naf[:, :, 9]) == 0

    def test_recent_moves_plane(self):
        """Plane 10 marks the last NUM_RECENTS moves."""
        g = fresh_game()
        moves = [Point(5, 5), Point(7, 7), Point(3, 3)]
        for m in moves:
            g.play(m)
        ni = NetInputs(g)
        # The 3 played moves should appear in plane 10 (WHITE to move now)
        # Check that recent move positions are marked
        # (plane 10 is from current player's perspective)
        assert numpy.count_nonzero(ni.naf[:, :, 10]) > 0

    def test_dtype_uint8(self):
        g = fresh_game()
        ni = NetInputs(g)
        assert ni.naf.dtype == numpy.uint8


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_compact_round_trip_empty(self):
        g = fresh_game()
        ni = NetInputs(g)
        b = ni.to_compact_bytes()
        assert len(b) == NetInputs.COMPACT_SIZE
        ni2 = NetInputs(b)
        assert ni.equal_with(ni2)

    def test_compact_round_trip_with_moves(self):
        g = fresh_game()
        g.play(Point(5, 5))
        g.play(Point(10, 10))
        g.play(Point(3, 3))
        ni = NetInputs(g)
        b = ni.to_compact_bytes()
        ni2 = NetInputs(b)
        assert ni.equal_with(ni2)

    def test_header_present(self):
        g = fresh_game()
        ni = NetInputs(g)
        b = ni.to_compact_bytes()
        assert b[:4] == b'JTwx'

    def test_wrong_length_raises(self):
        with pytest.raises((TypeError, ValueError, AssertionError)):
            NetInputs(b'tooshort')


# ---------------------------------------------------------------------------
# hflip / vflip symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    def _make_ni_with_moves(self):
        g = fresh_game()
        g.play(Point(5, 5))
        g.play(Point(10, 10))
        g.play(Point(3, 3))
        return NetInputs(g)

    def test_hflip_double_identity(self):
        ni = self._make_ni_with_moves()
        original_naf = ni.naf.copy()
        original_recents = list(ni.recents)
        ni.hflip()
        ni.hflip()
        assert numpy.array_equal(ni.naf, original_naf), "hflip x2 should be identity"
        assert ni.recents == original_recents

    def test_vflip_double_identity(self):
        ni = self._make_ni_with_moves()
        original_naf = ni.naf.copy()
        original_recents = list(ni.recents)
        ni.vflip()
        ni.vflip()
        assert numpy.array_equal(ni.naf, original_naf), "vflip x2 should be identity"
        assert ni.recents == original_recents

    def test_hflip_mirrors_recents_x(self):
        g = fresh_game()
        g.play(Point(5, 5))  # WHITE
        ni = NetInputs(g)
        recents_before = list(ni.recents)
        ni.hflip()
        for before, after in zip(recents_before, ni.recents):
            assert after.x == SIZE - 1 - before.x
            assert after.y == before.y

    def test_vflip_mirrors_recents_y(self):
        g = fresh_game()
        g.play(Point(5, 5))
        ni = NetInputs(g)
        recents_before = list(ni.recents)
        ni.vflip()
        for before, after in zip(recents_before, ni.recents):
            assert after.x == before.x
            assert after.y == SIZE - 1 - before.y


# ---------------------------------------------------------------------------
# Policy array symmetry helpers
# ---------------------------------------------------------------------------

class TestPolicyArraySymmetry:
    def test_hflip_policy_double_identity(self):
        arr = numpy.arange(NUM_MOVES, dtype=numpy.float32)
        arr2 = hflip_policy_array(hflip_policy_array(arr))
        assert numpy.array_equal(arr, arr2)

    def test_vflip_policy_double_identity(self):
        arr = numpy.arange(NUM_MOVES, dtype=numpy.float32)
        arr2 = vflip_policy_array(vflip_policy_array(arr))
        assert numpy.array_equal(arr, arr2)

    def test_single_move_policy_array(self):
        g = fresh_game()
        pt = Point(5, 5)
        arr = single_move_policy_array(g, pt)
        assert arr.shape == (NUM_MOVES,)
        idx = policy_point_index(g, pt)
        assert arr[idx] == 1
        assert numpy.sum(arr) == 1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_three_to_one_balanced(self):
        # Equal logits → score near 0
        score = three_to_one((0, 0, 0))
        assert abs(score) < 1e-9

    def test_three_to_one_winning(self):
        # Large win logit → score near +1
        score = three_to_one((-10, 0, 10))
        assert score > 0.99

    def test_three_to_one_losing(self):
        # Large loss logit → score near -1
        score = three_to_one((10, 0, -10))
        assert score < -0.99

    def test_three_to_one_range(self):
        for vals in [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0),
                     (-5, 0, 5), (5, 0, -5)]:
            score = three_to_one(vals)
            assert -1.0 <= score <= 1.0

    def test_one_to_three_win(self):
        assert one_to_three(1) == (0, 0, 1)

    def test_one_to_three_draw(self):
        assert one_to_three(0) == (0, 1, 0)

    def test_one_to_three_loss(self):
        assert one_to_three(-1) == (1, 0, 0)

    def test_location_inputs_shape(self):
        locs = location_inputs()
        assert locs.shape == (SIZE, SIZE, 2)

    def test_location_inputs_range(self):
        locs = location_inputs()
        assert locs.min() >= 0.0
        assert locs.max() <= 1.0

    def test_location_inputs_x_ramp(self):
        locs = location_inputs()
        # locs[:,:,1] = b.T where b[i,j]=a[i], so it increases along axis 0 (x/row)
        assert locs[0, 0, 1] < locs[SIZE // 2, 0, 1]
        assert locs[SIZE // 2, 0, 1] < locs[SIZE - 1, 0, 1]

    def test_location_inputs_y_ramp(self):
        locs = location_inputs()
        # locs[:,:,0] = b where b[i,j]=a[j], so it increases along axis 1 (y/col)
        assert locs[0, 0, 0] < locs[0, SIZE // 2, 0]
        assert locs[0, SIZE // 2, 0] < locs[0, SIZE - 1, 0]
