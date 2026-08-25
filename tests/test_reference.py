import numpy as np
import pytest

from nimmt.cards import N_CARDS, TOTAL_BULLHEADS
from nimmt.reference import RefGame, new_game, play_deal, play_turn, target_row


def lowest_row_policy(g, player, card):
    """牛頭が最小の行を選ぶ。同点なら index が小さい方。"""
    from nimmt.cards import BULL

    costs = [(sum(int(BULL[c]) for c in row), i) for i, row in enumerate(g.rows)]
    return min(costs)[1]


def lowest_card_policy(g, player):
    return min(g.hands[player])


def test_new_game_deals_distinct_cards():
    rng = np.random.default_rng(0)
    deck = (rng.permutation(N_CARDS) + 1).tolist()
    g = new_game(deck, 4)
    seen = [c for h in g.hands for c in h] + [r[0] for r in g.rows]
    assert len(seen) == 44
    assert len(set(seen)) == 44
    assert all(1 <= c <= 104 for c in seen)
    assert all(h == sorted(h) for h in g.hands)
    assert g.taken == [0, 0, 0, 0]


def test_target_row_picks_closest_lower_end():
    rows = [[10], [20], [30], [40]]
    assert target_row(rows, 35) == 2
    assert target_row(rows, 41) == 3
    assert target_row(rows, 11) == 0
    assert target_row(rows, 5) is None


def test_sixth_card_takes_the_row():
    g = RefGame(rows=[[1, 2, 3, 4, 5], [50], [60], [70]], hands=[[6]], taken=[0])
    pts = play_turn(g, [6], lowest_row_policy)
    # 1+1+1+1+2 = 6 牛頭
    assert pts == [6]
    assert g.taken == [6]
    assert g.rows[0] == [6]


def test_card_below_all_rows_takes_chosen_row():
    g = RefGame(rows=[[10], [20], [30], [40]], hands=[[5]], taken=[0])
    pts = play_turn(g, [5], lowest_row_policy)
    # 行0 (牛頭3) 行1 (3) 行2 (3) 行3 (3) → 同点なので index 0 を選ぶ
    assert pts == [3]
    assert g.rows[0] == [5]


def test_resolution_is_by_ascending_card_not_seat_order():
    """カード6を出した席1が先に解決されるので、席1が5枚を引き取る。
    席順で解決していたら席0（カード7）が引き取ることになる。"""
    g = RefGame(rows=[[1, 2, 3, 4, 5], [50], [60], [70]], hands=[[7], [6]], taken=[0, 0])
    pts = play_turn(g, [7, 6], lowest_row_policy)
    assert pts == [0, 6]
    assert g.rows[0] == [6, 7]


def test_full_deal_conserves_bullheads():
    rng = np.random.default_rng(7)
    for _ in range(50):
        deck = (rng.permutation(N_CARDS) + 1).tolist()
        taken = play_deal(deck, 4, lowest_card_policy, lowest_row_policy)
        assert len(taken) == 4
        assert all(t >= 0 for t in taken)
        assert sum(taken) <= TOTAL_BULLHEADS


def test_hands_are_empty_after_ten_turns():
    rng = np.random.default_rng(3)
    deck = (rng.permutation(N_CARDS) + 1).tolist()
    g = new_game(deck, 4)
    for _ in range(10):
        plays = [lowest_card_policy(g, p) for p in range(4)]
        play_turn(g, plays, lowest_row_policy)
    assert all(len(h) == 0 for h in g.hands)
