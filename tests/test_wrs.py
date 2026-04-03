"""
Tests for wrs.py — WeightedRandomSelector.

Pure Python module; no neural net or game logic involved.
"""
import math
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wrs import WeightedRandomSelector


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_empty_selector(self):
        wrs = WeightedRandomSelector()
        assert wrs.total_weight == 0.0
        assert wrs.default_weight == 1.0

    def test_set_default_weight(self):
        wrs = WeightedRandomSelector()
        wrs.set_default_weight(2.5)
        assert wrs.default_weight == 2.5

    def test_set_default_weight_zero(self):
        wrs = WeightedRandomSelector()
        wrs.set_default_weight(0.0)
        assert wrs.default_weight == 0.0

    def test_set_default_weight_negative_raises(self):
        wrs = WeightedRandomSelector()
        with pytest.raises(AssertionError):
            wrs.set_default_weight(-1.0)


# ---------------------------------------------------------------------------
# add_basket
# ---------------------------------------------------------------------------

class TestAddBasket:
    def test_single_basket_updates_total_weight(self):
        wrs = WeightedRandomSelector()
        wrs.add_basket(10)
        assert wrs.total_weight == pytest.approx(10.0)

    def test_explicit_weight(self):
        wrs = WeightedRandomSelector()
        wrs.add_basket(5, weight=2.0)
        assert wrs.total_weight == pytest.approx(10.0)

    def test_two_baskets_accumulate(self):
        wrs = WeightedRandomSelector()
        wrs.add_basket(10)
        wrs.add_basket(20)
        assert wrs.total_weight == pytest.approx(30.0)

    def test_obj_stored(self):
        wrs = WeightedRandomSelector()
        sentinel = object()
        wrs.add_basket(5, obj=sentinel)
        assert wrs.item_objects[0] is sentinel

    def test_none_obj_by_default(self):
        wrs = WeightedRandomSelector()
        wrs.add_basket(5)
        assert wrs.item_objects[0] is None

    def test_uses_default_weight(self):
        wrs = WeightedRandomSelector()
        wrs.set_default_weight(3.0)
        wrs.add_basket(4)
        assert wrs.total_weight == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# random_item — return types
# ---------------------------------------------------------------------------

class TestRandomItem:
    def test_returns_three_values(self):
        wrs = WeightedRandomSelector()
        wrs.add_basket(10, obj='a')
        i, j, obj = wrs.random_item()
        assert isinstance(i, int)
        assert isinstance(j, int)

    def test_single_basket_always_returns_it(self):
        wrs = WeightedRandomSelector()
        wrs.add_basket(10, obj='only')
        for _ in range(20):
            i, j, obj = wrs.random_item()
            assert i == 0
            assert obj == 'only'

    def test_index_within_count(self):
        count = 7
        wrs = WeightedRandomSelector()
        wrs.add_basket(count)
        for _ in range(50):
            _, j, _ = wrs.random_item()
            assert 0 <= j < count

    def test_two_baskets_both_reachable(self):
        random.seed(42)
        wrs = WeightedRandomSelector()
        wrs.add_basket(10, obj='a')
        wrs.add_basket(10, obj='b')
        seen = set()
        for _ in range(200):
            _, _, obj = wrs.random_item()
            seen.add(obj)
        assert seen == {'a', 'b'}

    def test_zero_weight_basket_never_chosen(self):
        random.seed(0)
        wrs = WeightedRandomSelector()
        wrs.add_basket(100, obj='heavy', weight=1.0)
        wrs.add_basket(100, obj='zero', weight=0.0)
        for _ in range(100):
            _, _, obj = wrs.random_item()
            assert obj == 'heavy'


# ---------------------------------------------------------------------------
# random_item — statistical distribution
# ---------------------------------------------------------------------------

class TestDistribution:
    def _run_n(self, wrs, n, seed=0):
        """Return Counter of (basket_index, item_index) pairs."""
        from collections import Counter
        rng_state = random.getstate()
        random.seed(seed)
        ctr = Counter()
        for _ in range(n):
            i, j, obj = wrs.random_item()
            ctr[(i, j)] += 1
        random.setstate(rng_state)
        return ctr

    def test_uniform_within_basket(self):
        """Items within a single basket should be roughly uniformly distributed."""
        count = 5
        wrs = WeightedRandomSelector()
        wrs.add_basket(count)
        N = 50000
        ctr = self._run_n(wrs, N)
        expected = N / count
        for j in range(count):
            actual = ctr[(0, j)]
            # Allow 10% deviation
            assert abs(actual - expected) / expected < 0.10, \
                f"Item {j} got {actual} vs expected {expected:.0f}"

    def test_basket_weight_proportional(self):
        """Heavier basket should get proportionally more hits."""
        wrs = WeightedRandomSelector()
        wrs.add_basket(1, obj='light', weight=1.0)
        wrs.add_basket(1, obj='heavy', weight=3.0)
        N = 40000
        from collections import Counter
        random.seed(7)
        ctr = Counter()
        for _ in range(N):
            i, j, obj = wrs.random_item()
            ctr[obj] += 1
        ratio = ctr['heavy'] / ctr['light']
        assert 2.5 < ratio < 3.5, f"Expected ratio ~3, got {ratio:.2f}"

    def test_total_calls_add_up(self):
        wrs = WeightedRandomSelector()
        wrs.add_basket(5, obj='x')
        wrs.add_basket(3, obj='y')
        N = 100
        ctr = self._run_n(wrs, N)
        assert sum(ctr.values()) == N
