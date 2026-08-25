import numpy as np
import pytest

from nimmt.cards import HAND_SIZE, N_CARDS, TOTAL_BULLHEADS
from nimmt.reference import play_deal
from nimmt.vec import PHASE_CARD, PHASE_ROW, VecNimmt, new_decks


def lowest_card_slots(env):
    """手札の最小カードのスロット index。(B,P)"""
    h = np.where(env.hands > 0, env.hands, np.int16(32767))
    return h.argmin(axis=2)


def lowest_bull_rows(env):
    """牛頭最小の行。同点は index 最小（参照実装の min() と一致させる）。"""
    return env.row_points_all(env.pending_games).argmin(axis=1)


def run_vec_deal(env):
    while not env.done():
        env.step_cards(lowest_card_slots(env))
        while env.phase == PHASE_ROW:
            env.step_rows(lowest_bull_rows(env))
    return env.taken.copy()


def test_new_decks_are_permutations():
    rng = np.random.default_rng(0)
    d = new_decks(16, rng)
    assert d.shape == (16, N_CARDS)
    assert d.dtype == np.int16
    for row in d:
        assert sorted(row.tolist()) == list(range(1, N_CARDS + 1))


def test_reset_shapes_and_dtypes():
    env = VecNimmt(8, 4, seed=1)
    assert env.rows.shape == (8, 4, 5) and env.rows.dtype == np.int16
    assert env.row_len.shape == (8, 4) and env.row_len.dtype == np.int8
    assert env.hands.shape == (8, 4, 10) and env.hands.dtype == np.int16
    assert env.taken.shape == (8, 4) and env.taken.dtype == np.int16
    assert env.seen.shape == (8, 105) and env.seen.dtype == np.bool_
    assert (env.row_len == 1).all()
    assert env.turn == 0 and env.phase == PHASE_CARD
    # 各ゲームで 44 枚が相異なる
    for b in range(8):
        cards = env.hands[b].ravel().tolist() + env.rows[b, :, 0].tolist()
        assert len(set(cards)) == 44
    # 手札は昇順
    assert (np.diff(env.hands, axis=2) > 0).all()


def test_seen_starts_with_row_cards_only():
    env = VecNimmt(4, 4, seed=2)
    assert env.seen.sum(axis=1).tolist() == [4, 4, 4, 4]
    for b in range(4):
        for c in env.rows[b, :, 0]:
            assert env.seen[b, c]


def test_sixth_card_takes_the_row():
    env = VecNimmt(1, 1, seed=0)
    env.rows[:] = 0
    env.rows[0, 0, :] = [1, 2, 3, 4, 5]
    env.rows[0, 1, 0], env.rows[0, 2, 0], env.rows[0, 3, 0] = 50, 60, 70
    env.row_len[0] = [5, 1, 1, 1]
    env.hands[0, 0, :] = 0
    env.hands[0, 0, 0] = 6
    env.step_cards(np.zeros((1, 1), dtype=np.intp))
    assert env.phase == PHASE_CARD
    assert env.taken[0, 0] == 6  # 1+1+1+1+2
    assert env.row_len[0, 0] == 1 and env.rows[0, 0, 0] == 6


def test_card_below_all_rows_requests_row_choice():
    env = VecNimmt(1, 1, seed=0)
    env.rows[:] = 0
    env.rows[0, :, 0] = [10, 20, 30, 40]
    env.row_len[0] = [1, 1, 1, 1]
    env.hands[0, 0, :] = 0
    env.hands[0, 0, 0] = 5
    env.step_cards(np.zeros((1, 1), dtype=np.intp))
    assert env.phase == PHASE_ROW
    assert env.pending_games.tolist() == [0]
    assert env.pending_cards.tolist() == [5]
    env.step_rows(np.array([2]))
    assert env.phase == PHASE_CARD
    assert env.taken[0, 0] == 3  # 30 の牛頭
    assert env.rows[0, 2, 0] == 5 and env.row_len[0, 2] == 1


def test_resolution_is_by_ascending_card_not_seat_order():
    env = VecNimmt(1, 2, seed=0)
    env.rows[:] = 0
    env.rows[0, 0, :] = [1, 2, 3, 4, 5]
    env.rows[0, 1, 0], env.rows[0, 2, 0], env.rows[0, 3, 0] = 50, 60, 70
    env.row_len[0] = [5, 1, 1, 1]
    env.hands[0] = 0
    env.hands[0, 0, 0] = 7  # 席0
    env.hands[0, 1, 0] = 6  # 席1
    env.step_cards(np.zeros((1, 2), dtype=np.intp))
    assert env.taken[0].tolist() == [0, 6]
    assert env.rows[0, 0, :2].tolist() == [6, 7]
    assert env.row_len[0, 0] == 2


def test_row_points_all():
    env = VecNimmt(1, 1, seed=0)
    env.rows[:] = 0
    env.rows[0, 0, :2] = [10, 11]   # 3 + 5 = 8
    env.rows[0, 1, 0] = 55          # 7
    env.rows[0, 2, 0] = 1           # 1
    env.rows[0, 3, 0] = 5           # 2
    env.row_len[0] = [2, 1, 1, 1]
    assert env.row_points_all(np.array([0]))[0].tolist() == [8, 7, 1, 2]


@pytest.mark.parametrize("n_players", [2, 4])
def test_matches_reference_implementation(n_players):
    """同じ配牌・同じ決定的方策で、参照実装と失点が完全一致すること。"""
    n_deals = 5000
    rng = np.random.default_rng(1234)
    decks = new_decks(n_deals, rng)

    env = VecNimmt(n_deals, n_players, seed=0)
    env.reset(decks)
    vec_taken = run_vec_deal(env)

    from nimmt.cards import BULL

    def ref_card(g, p):
        return min(g.hands[p])

    def ref_row(g, p, card):
        return min((sum(int(BULL[c]) for c in row), i) for i, row in enumerate(g.rows))[1]

    ref_taken = np.array(
        [play_deal(decks[i].tolist(), n_players, ref_card, ref_row) for i in range(n_deals)],
        dtype=np.int16,
    )
    mismatch = np.flatnonzero((vec_taken != ref_taken).any(axis=1))
    assert mismatch.size == 0, f"{mismatch.size} deals differ, first={mismatch[:5]}"


def test_deal_terminates_and_conserves_bullheads():
    env = VecNimmt(256, 4, seed=5)
    taken = run_vec_deal(env)
    assert env.turn == HAND_SIZE
    assert (env.hands == 0).all()
    assert (taken >= 0).all()
    assert (taken.sum(axis=1) <= TOTAL_BULLHEADS).all()
