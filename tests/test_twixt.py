"""
Tests for twixt.py — TwixT game logic.

Reference baseline for the Python 2 → Python 3 / PyTorch migration.
All tests must pass after minimal syntax fixes to twixt.py; no logic changes.
"""
import numpy
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import twixt
from twixt import Point, SelectSet, Game


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_game():
    return Game()


def game_snapshot(g):
    """Return a dict capturing the full mutable state of a Game for comparison."""
    return {
        'pegs': [p.copy() for p in g.pegs],
        'links': [lk.copy() for lk in g.links],
        'turn': g.turn,
        'history_len': len(g.history),
        'white_open': set(g.open_pegs[Game.WHITE].item_by_index),
        'black_open': set(g.open_pegs[Game.BLACK].item_by_index),
        'white_reachable': frozenset(g.reachable[Game.WHITE]),
        'black_reachable': frozenset(g.reachable[Game.BLACK]),
    }


def snapshots_equal(a, b):
    for key in ('turn', 'history_len', 'white_open', 'black_open',
                'white_reachable', 'black_reachable'):
        if a[key] != b[key]:
            return False, f"mismatch on {key}: {a[key]!r} vs {b[key]!r}"
    for i in range(2):
        if not numpy.array_equal(a['pegs'][i], b['pegs'][i]):
            return False, f"pegs[{i}] differ"
    for i in range(8):
        if not numpy.array_equal(a['links'][i], b['links'][i]):
            return False, f"links[{i}] differ"
    return True, "ok"


# ---------------------------------------------------------------------------
# Point class
# ---------------------------------------------------------------------------

class TestPoint:
    def test_construct_xy(self):
        p = Point(3, 7)
        assert p.x == 3
        assert p.y == 7

    def test_construct_string_lowercase(self):
        p = Point("a1")
        assert p.x == 0
        assert p.y == 0

    def test_construct_string_b3(self):
        p = Point("b3")
        assert p.x == 1
        assert p.y == 2

    def test_construct_string_uppercase(self):
        p = Point("A1")
        assert p.x == 0
        assert p.y == 0

    def test_str_round_trip(self):
        assert str(Point(0, 0)) == "a1"
        assert str(Point(1, 2)) == "b3"
        assert str(Point(23, 23)) == "x24"

    def test_flip(self):
        assert Point(3, 5).flip() == Point(5, 3)
        assert Point(0, 0).flip() == Point(0, 0)
        assert Point(7, 2).flip() == Point(2, 7)

    def test_add_points(self):
        assert Point(1, 2) + Point(3, 4) == Point(4, 6)
        assert Point(0, 0) + Point(5, 5) == Point(5, 5)

    def test_sub_points(self):
        assert Point(3, 4) - Point(1, 2) == Point(2, 2)
        assert Point(5, 5) - Point(5, 5) == Point(0, 0)

    def test_add_tuple(self):
        assert Point(1, 2) + (3, 4) == Point(4, 6)

    def test_sub_tuple(self):
        assert Point(3, 4) - (1, 2) == Point(2, 2)

    def test_mul_scalar(self):
        assert Point(2, 3) * 2 == Point(4, 6)
        assert 3 * Point(1, 2) == Point(3, 6)

    def test_construct_from_sequence(self):
        p = Point([5, 7])
        assert p.x == 5
        assert p.y == 7

    def test_bad_construct_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Point("bad")


# ---------------------------------------------------------------------------
# SelectSet
# ---------------------------------------------------------------------------

class TestSelectSet:
    def test_add_contains_len(self):
        s = SelectSet()
        s.add(10)
        s.add(20)
        assert 10 in s
        assert 20 in s
        assert 99 not in s
        assert len(s) == 2

    def test_remove_last(self):
        s = SelectSet()
        s.add(1)
        s.add(2)
        s.remove(2)
        assert 2 not in s
        assert 1 in s
        assert len(s) == 1

    def test_remove_middle_swaps(self):
        s = SelectSet()
        for i in range(5):
            s.add(i)
        s.remove(2)
        assert 2 not in s
        assert len(s) == 4
        for i in (0, 1, 3, 4):
            assert i in s

    def test_duplicate_add_raises(self):
        s = SelectSet()
        s.add(42)
        with pytest.raises(ValueError):
            s.add(42)

    def test_getitem(self):
        s = SelectSet()
        s.add("a")
        s.add("b")
        assert s[0] == "a"
        assert s[1] == "b"

    def test_clone_independence(self):
        s = SelectSet()
        s.add(1)
        s.add(2)
        c = s.clone()
        # add to clone — original unchanged
        c.add(3)
        assert 3 not in s
        # add to original — clone unchanged
        s.add(99)
        assert 99 not in c


# ---------------------------------------------------------------------------
# Game initialisation
# ---------------------------------------------------------------------------

class TestGameInit:
    def test_turn_is_white(self):
        assert fresh_game().turn == Game.WHITE

    def test_empty_history(self):
        assert len(fresh_game().history) == 0

    def test_pegs_all_zero(self):
        g = fresh_game()
        for i in range(2):
            assert numpy.all(g.pegs[i] == 0)

    def test_links_all_zero(self):
        g = fresh_game()
        for i in range(8):
            assert numpy.all(g.links[i] == 0)

    def test_white_open_count(self):
        # WHITE can play on columns 1..22 (22 cols) × rows 0..23 (24 rows) = 528
        assert len(fresh_game().open_pegs[Game.WHITE]) == 22 * 24

    def test_black_open_count(self):
        # BLACK can play on rows 1..22 (22 rows) × cols 0..23 (24 cols) = 528
        assert len(fresh_game().open_pegs[Game.BLACK]) == 22 * 24

    def test_not_winning_initially(self):
        g = fresh_game()
        assert g.is_winning(Game.WHITE) is False
        assert g.is_winning(Game.BLACK) is False


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------

class TestLegalMoves:
    def test_legal_matches_slow_white(self):
        g = fresh_game()
        fast = set(g.open_pegs[Game.WHITE].item_by_index)
        slow = set(p for p in g.slow_legal_plays() if isinstance(p, Point))
        assert fast == slow

    def test_legal_matches_slow_black(self):
        g = fresh_game()
        g.play(Point(5, 5))  # WHITE plays, now BLACK's turn
        fast = set(g.open_pegs[Game.BLACK].item_by_index)
        slow = set(p for p in g.slow_legal_plays() if isinstance(p, Point))
        assert fast == slow

    def test_white_cannot_play_col0(self):
        g = fresh_game()
        assert Point(0, 5) not in g.open_pegs[Game.WHITE]

    def test_white_cannot_play_col23(self):
        g = fresh_game()
        assert Point(23, 5) not in g.open_pegs[Game.WHITE]

    def test_black_cannot_play_row0(self):
        g = fresh_game()
        g.play(Point(5, 5))
        assert Point(5, 0) not in g.open_pegs[Game.BLACK]

    def test_black_cannot_play_row23(self):
        g = fresh_game()
        g.play(Point(5, 5))
        assert Point(5, 23) not in g.open_pegs[Game.BLACK]

    def test_played_cell_removed_from_both(self):
        g = fresh_game()
        p = Point(5, 5)
        g.play(p)
        assert p not in g.open_pegs[Game.WHITE]
        assert p not in g.open_pegs[Game.BLACK]

    def test_undo_restores_cell(self):
        g = fresh_game()
        p = Point(5, 5)
        g.play(p)
        g.undo()
        assert p in g.open_pegs[Game.WHITE]
        assert p in g.open_pegs[Game.BLACK]

    def test_inbounds_for_player_white(self):
        # WHITE valid: x in 1..22, y in 0..23
        assert Game.inbounds_for_player(Point(1, 0), Game.WHITE)
        assert Game.inbounds_for_player(Point(22, 23), Game.WHITE)
        assert not Game.inbounds_for_player(Point(0, 5), Game.WHITE)
        assert not Game.inbounds_for_player(Point(23, 5), Game.WHITE)

    def test_inbounds_for_player_black(self):
        # BLACK valid: x in 0..23, y in 1..22
        assert Game.inbounds_for_player(Point(0, 1), Game.BLACK)
        assert Game.inbounds_for_player(Point(23, 22), Game.BLACK)
        assert not Game.inbounds_for_player(Point(5, 0), Game.BLACK)
        assert not Game.inbounds_for_player(Point(5, 23), Game.BLACK)


# ---------------------------------------------------------------------------
# inbounds static helper
# ---------------------------------------------------------------------------

class TestInbounds:
    def test_corners(self):
        assert Game.inbounds(Point(0, 0))
        assert Game.inbounds(Point(23, 23))
        assert Game.inbounds(Point(0, 23))
        assert Game.inbounds(Point(23, 0))

    def test_center(self):
        assert Game.inbounds(Point(12, 12))

    def test_out_of_bounds(self):
        assert not Game.inbounds(Point(-1, 0))
        assert not Game.inbounds(Point(0, -1))
        assert not Game.inbounds(Point(24, 0))
        assert not Game.inbounds(Point(0, 24))


# ---------------------------------------------------------------------------
# Play / undo symmetry (most critical)
# ---------------------------------------------------------------------------

class TestPlayUndoSymmetry:
    def test_single_move(self):
        g = fresh_game()
        snap0 = game_snapshot(g)
        g.play(Point(5, 5))
        g.undo()
        snap1 = game_snapshot(g)
        eq, msg = snapshots_equal(snap0, snap1)
        assert eq, msg

    def test_multiple_moves_full_undo(self):
        g = fresh_game()
        snap0 = game_snapshot(g)
        moves = [Point(5, 5), Point(7, 6), Point(3, 3), Point(10, 10), Point(12, 1)]
        for m in moves:
            g.play(m)
        for _ in moves:
            g.undo()
        snap1 = game_snapshot(g)
        eq, msg = snapshots_equal(snap0, snap1)
        assert eq, msg

    def test_linking_move_undo(self):
        """Place two WHITE pegs a knight's move apart so a link forms, then undo."""
        g = fresh_game()
        a = Point(5, 5)
        b = Point(7, 6)   # knight's move from a: dx=2, dy=1
        snap0 = game_snapshot(g)
        g.play(a)
        # now BLACK's turn; play a BLACK move that doesn't interfere
        g.play(Point(1, 1))
        # now WHITE again
        g.play(b)
        # link should exist
        assert g.get_link(a, b, Game.WHITE) == 1
        g.undo()  # undo b
        assert g.get_link(a, b, Game.WHITE) == 0
        g.undo()  # undo black
        g.undo()  # undo a
        snap1 = game_snapshot(g)
        eq, msg = snapshots_equal(snap0, snap1)
        assert eq, msg

    def test_turn_alternates_correct(self):
        g = fresh_game()
        assert g.turn == Game.WHITE
        g.play(Point(5, 5))
        assert g.turn == Game.BLACK
        g.play(Point(7, 7))
        assert g.turn == Game.WHITE
        g.undo()
        assert g.turn == Game.BLACK
        g.undo()
        assert g.turn == Game.WHITE

    def test_reachable_consistent_after_undo(self):
        """is_winning and slow_is_winning agree after play+undo."""
        g = fresh_game()
        g.play(Point(5, 5))
        g.play(Point(5, 6))
        g.undo()
        g.undo()
        assert g.is_winning(Game.WHITE) == g.slow_is_winning(Game.WHITE)
        assert g.is_winning(Game.BLACK) == g.slow_is_winning(Game.BLACK)


# ---------------------------------------------------------------------------
# Auto-link placement
# ---------------------------------------------------------------------------

class TestAutoLink:
    def _white_peg_at(self, g, p):
        """Force a WHITE peg without going through play() — for setup only."""
        g.pegs[Game.WHITE][p] = 1
        g.open_pegs[Game.WHITE].remove(p)
        g.open_pegs[Game.BLACK].remove(p)

    def test_link_placed_between_two_white_pegs(self):
        """Play WHITE at two knight-move-apart cells; link should form."""
        g = fresh_game()
        a = Point(5, 5)
        b = Point(7, 6)  # dx=2, dy=1 — valid DLINK
        g.play(a)
        # BLACK's turn; play somewhere neutral
        g.play(Point(1, 1))
        # WHITE plays at b
        g.play(b)
        assert g.get_link(a, b, Game.WHITE) == 1, "Link should have formed"

    def test_all_dlink_directions_create_links(self):
        """For each of the 8 knight-move directions, a link should form."""
        for dlink in Game.DLINKS:
            g = fresh_game()
            a = Point(12, 12)
            b = a + dlink
            if not Game.inbounds(b):
                continue
            if not Game.inbounds_for_player(a, Game.WHITE):
                continue
            if not Game.inbounds_for_player(b, Game.WHITE):
                continue
            g.play(a)
            g.play(Point(1, 1))  # BLACK's turn
            g.play(b)
            assert g.get_link(a, b, Game.WHITE) == 1, \
                f"Link not formed for dlink {dlink}: {a} → {b}"

    def test_link_blocked_by_crossing_enemy_link(self):
        """A link that would cross a BLACK link should not be placed."""
        g = fresh_game()
        # Place WHITE pegs at (5,5) and (7,6) (dx=2,dy=1 knight move)
        # We'll manually place a BLACK link that crosses this path.
        # The crossing link for (5,5)→(7,6) should be a BLACK link
        # that geometrically intersects it.
        # Crossing link: BLACK peg-pair (5,6)→(7,5) — dx=2, dy=-1 crosses dx=2,dy=1
        a_white = Point(5, 5)
        b_white = Point(7, 6)
        a_black = Point(5, 6)
        b_black = Point(7, 5)

        # Manually set the crossing BLACK link (bypassing play to avoid turn issues)
        g.set_link(a_black, b_black, Game.BLACK, 1)
        g.pegs[Game.BLACK][a_black] = 1
        g.pegs[Game.BLACK][b_black] = 1

        # Now play WHITE at a_white (using play())
        g.play(a_white)
        # BLACK's turn: skip with a neutral move
        g.play(Point(1, 1))
        # WHITE plays b_white — link should be blocked
        g.play(b_white)
        assert g.get_link(a_white, b_white, Game.WHITE) == 0, \
            "Link should be blocked by crossing BLACK link"


# ---------------------------------------------------------------------------
# any_crossing_links
# ---------------------------------------------------------------------------

class TestAnyCrossingLinks:
    def test_no_links_no_crossing(self):
        g = fresh_game()
        # No links on board; any_crossing_links should return False
        assert not g.any_crossing_links(Point(5, 5), Point(7, 6), Game.BLACK)

    def test_crossing_link_detected(self):
        """A BLACK link crossing the WHITE path should be detected."""
        g = fresh_game()
        a = Point(5, 5)
        b = Point(7, 6)
        # Place a BLACK link that crosses a→b: (5,6)→(7,5)
        g.set_link(Point(5, 6), Point(7, 5), Game.BLACK, 1)
        assert g.any_crossing_links(a, b, Game.BLACK)

    def test_parallel_link_not_crossing(self):
        """A parallel BLACK link should not block."""
        g = fresh_game()
        a = Point(5, 5)
        b = Point(7, 6)
        # Parallel link far away
        g.set_link(Point(10, 10), Point(12, 11), Game.BLACK, 1)
        assert not g.any_crossing_links(a, b, Game.BLACK)

    def test_do_links_cross_crossing_pair(self):
        # Use links with different |slopes| so do_links_cross works.
        # (3,5)→(5,4) slope=-0.5 crosses (4,3)→(5,5) slope=2
        linka = (Point(3, 5), Point(5, 4))
        linkb = (Point(4, 3), Point(5, 5))
        assert Game.do_links_cross(linka, linkb)

    def test_do_links_cross_parallel(self):
        linka = (Point(5, 5), Point(7, 6))
        linkb = (Point(5, 7), Point(7, 8))
        assert not Game.do_links_cross(linka, linkb)

    def test_do_links_cross_same_slope(self):
        # Same |slope| → not crossing (parallel diagonals)
        linka = (Point(0, 0), Point(2, 1))
        linkb = (Point(4, 0), Point(6, 1))
        assert not Game.do_links_cross(linka, linkb)


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------

class TestWinDetection:
    def _build_white_chain(self, g):
        """Build a WHITE chain spanning y=0 to y=23 via a staircase of knight moves.

        Path: (5,0) → (7,1) → (5,2) → (7,3) → ... zigzag pattern.
        WHITE cannot play on x=0 or x=23, so stay in x=5 and x=7.
        Each move alternates WHITE/BLACK; we skip BLACK's turn with neutral moves.
        """
        # Build the chain: alternating x=5 and x=7, stepping y by 1 each time
        # via the knight move (±2, ±1) or (±1, ±2)
        # Simplest spanning path: column x=5, every 2 rows, linked via (5,y)→(7,y+1)→(5,y+2)
        chain = []
        x = 5
        for y in range(0, 24):
            chain.append(Point(x, y))
            x = 12 - x  # alternates between 5 and 7

        for i, pt in enumerate(chain):
            g.play(pt)  # WHITE
            if i < len(chain) - 1:
                # BLACK plays a neutral cell (col 15, cycling through rows)
                bpt = Point(15, i % 22 + 1)
                if bpt not in g.open_pegs[Game.BLACK]:
                    bpt = Point(16, i % 22 + 1)
                g.play(bpt)

        return chain

    def test_empty_board_not_winning(self):
        g = fresh_game()
        assert not g.is_winning(Game.WHITE)
        assert not g.is_winning(Game.BLACK)
        assert g.is_winning(Game.WHITE) == g.slow_is_winning(Game.WHITE)
        assert g.is_winning(Game.BLACK) == g.slow_is_winning(Game.BLACK)

    def test_white_wins_with_chain(self):
        g = fresh_game()
        self._build_white_chain(g)
        assert g.is_winning(Game.WHITE)
        # Consistency check
        assert g.is_winning(Game.WHITE) == g.slow_is_winning(Game.WHITE)

    def test_near_miss_not_winning(self):
        """WHITE chain from y=0 to y=22 (not y=23) → not winning."""
        g = fresh_game()
        # Place WHITE pegs in a column that doesn't reach y=23
        # Use a simple vertical-ish path capped at y=22
        x = 5
        for y in range(0, 23):  # 0..22, stop before 23
            pt = Point(x, y)
            g.play(pt)
            x2 = 12 - x
            if y < 22:
                bpt = Point(15, y + 1)
                if bpt not in g.open_pegs[Game.BLACK]:
                    bpt = Point(16, y + 1)
                g.play(bpt)
            x = x2

        assert not g.is_winning(Game.WHITE)

    def test_win_then_undo_not_winning(self):
        """After winning, undo the last move → no longer winning."""
        g = fresh_game()
        chain = self._build_white_chain(g)
        assert g.is_winning(Game.WHITE)
        # Undo WHITE's last move (which was the winning move)
        # We need to also undo the last BLACK move that followed
        # Actually _build_white_chain plays WHITE then BLACK alternately;
        # the last iteration plays WHITE with no following BLACK.
        g.undo()
        assert not g.is_winning(Game.WHITE)

    def test_just_won(self):
        """just_won() reflects the player who just moved."""
        g = fresh_game()
        self._build_white_chain(g)
        # After the chain, it's BLACK's turn (WHITE just won)
        assert g.just_won()

    def test_is_winning_black(self):
        """BLACK wins by spanning x=0 to x=23."""
        g = fresh_game()
        # BLACK chain: zigzag across y=10 and y=12 spanning all 24 x-columns.
        # WHITE plays 23 filler moves; precompute safe cells (columns 15-17, rows 3-8)
        # that don't overlap with the BLACK chain positions.
        black_chain = []
        y = 10
        for x in range(24):
            black_chain.append(Point(x, y))
            y = 22 - y  # alternates 10 ↔ 12

        white_pool = [Point(col, row)
                      for col in (15, 16, 17)
                      for row in range(1, 23)
                      if Point(col, row) not in black_chain]

        # WHITE always moves first; play WHITE before each BLACK move
        white_idx = 0
        g.play(white_pool[white_idx]); white_idx += 1  # WHITE first
        for i, pt in enumerate(black_chain):
            g.play(pt)   # BLACK
            if i < len(black_chain) - 1:
                g.play(white_pool[white_idx]); white_idx += 1  # WHITE

        assert g.is_winning(Game.BLACK)
        assert g.is_winning(Game.BLACK) == g.slow_is_winning(Game.BLACK)


# ---------------------------------------------------------------------------
# Clone isolation
# ---------------------------------------------------------------------------

class TestClone:
    def test_clone_independent_play(self):
        g = fresh_game()
        g.play(Point(5, 5))
        c = g.clone()
        # Play more on clone
        c.play(Point(7, 6))
        # Original should be unchanged
        assert Point(7, 6) in g.open_pegs[Game.BLACK]  # not played in original

    def test_clone_peg_array_independent(self):
        g = fresh_game()
        c = g.clone()
        c.pegs[Game.WHITE][5, 5] = 1
        assert g.pegs[Game.WHITE][5, 5] == 0

    def test_clone_has_same_state(self):
        g = fresh_game()
        g.play(Point(5, 5))
        g.play(Point(10, 10))
        c = g.clone()
        snap_g = game_snapshot(g)
        snap_c = game_snapshot(c)
        eq, msg = snapshots_equal(snap_g, snap_c)
        assert eq, msg

    def test_clone_reachable_independent(self):
        g = fresh_game()
        c = g.clone()
        c.reachable[Game.WHITE].add("something")
        assert "something" not in g.reachable[Game.WHITE]
