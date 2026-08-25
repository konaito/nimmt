import numpy as np
from nimmt.cards import BULL, N_CARDS, TOTAL_BULLHEADS, bullheads


def test_bullhead_values():
    assert bullheads(55) == 7
    for c in (11, 22, 33, 44, 66, 77, 88, 99):
        assert bullheads(c) == 5
    for c in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        assert bullheads(c) == 3
    for c in (5, 15, 25, 35, 45, 65, 75, 85, 95):
        assert bullheads(c) == 2
    for c in (1, 2, 3, 4, 6, 7, 101, 102, 103, 104):
        assert bullheads(c) == 1


def test_distribution_matches_published_counts():
    counts = {}
    for c in range(1, N_CARDS + 1):
        counts[bullheads(c)] = counts.get(bullheads(c), 0) + 1
    # Wikipedia / 公式ルールの分布
    assert counts == {7: 1, 5: 8, 3: 10, 2: 9, 1: 76}
    assert sum(k * v for k, v in counts.items()) == TOTAL_BULLHEADS == 171


def test_bull_table_matches_function():
    assert BULL.shape == (105,)
    assert BULL.dtype == np.int16
    assert BULL[0] == 0
    for c in range(1, N_CARDS + 1):
        assert BULL[c] == bullheads(c)
