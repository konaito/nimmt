import numpy as np

from nimmt.bots.base import argmin_masked, min_bull_rows
from nimmt.bots.simple import GreedyBot, HeuristicBot, RandomBot
from nimmt.features import placement_info
from nimmt.vec import PHASE_ROW, VecNimmt, new_decks

ALL_BOTS = [RandomBot(seed=0), GreedyBot(), HeuristicBot()]


def play_all_seats(env, bot):
    games = np.repeat(np.arange(env.B), env.P)
    players = np.tile(np.arange(env.P), env.B)
    while not env.done():
        slots = bot.pick_cards(env, games, players).reshape(env.B, env.P)
        env.step_cards(slots)
        while env.phase == PHASE_ROW:
            env.step_rows(bot.pick_rows(env, np.arange(env.pending_games.size)))
    return env.taken.copy()


def test_argmin_masked_ignores_invalid():
    cost = np.array([[5.0, 1.0, 9.0]])
    mask = np.array([[True, False, True]])
    assert argmin_masked(cost, mask).tolist() == [0]


def test_placement_info_raw_values():
    env = VecNimmt(1, 2, seed=0)
    env.rows[:] = 0
    env.rows[0, 0, :2] = [10, 11]
    env.rows[0, 1, 0] = 20
    env.rows[0, 2, 0] = 30
    env.rows[0, 3, :] = [40, 41, 42, 43, 44]   # 牛頭 3+1+1+1+5 = 11（44 は11の倍数）
    env.row_len[0] = [2, 1, 1, 5]
    env.hands[0] = 0
    env.hands[0, 0, :3] = [5, 12, 45]
    info = placement_info(env, np.array([0]), np.array([0]))
    assert info["below_all"][0, :3].tolist() == [True, False, False]
    assert info["is_sixth"][0, :3].tolist() == [False, False, True]
    assert info["take_now"][0, :3].tolist() == [3.0, 0.0, 11.0]
    assert info["gap"][0, 1] == 1.0
    assert info["min_bull"][0] == 3.0


def test_greedy_avoids_taking_when_free_option_exists():
    env = VecNimmt(1, 1, seed=0)
    env.rows[:] = 0
    env.rows[0, 0, :] = [40, 41, 42, 43, 44]   # 6枚目を踏むと牛頭11（44 は11の倍数）
    env.rows[0, 1, 0] = 10
    env.rows[0, 2, 0] = 20
    env.rows[0, 3, 0] = 30
    env.row_len[0] = [5, 1, 1, 1]
    env.hands[0] = 0
    env.hands[0, 0, :2] = [31, 45]             # 31 は無傷、45 は牛頭11
    slot = GreedyBot().pick_cards(env, np.array([0]), np.array([0]))
    assert slot.tolist() == [0]


def test_min_bull_rows_prefers_fewest_cards_on_tie():
    """牛頭が同点なら枚数が少ない行を選ぶ。index が先の行より優先されること。"""
    env = VecNimmt(1, 1, seed=0)
    env.rows[:] = 0
    env.rows[0, 0, :2] = [1, 2]        # 牛頭 1+1 = 2、2枚
    env.rows[0, 1, 0] = 5              # 牛頭 2、1枚
    env.rows[0, 2, :2] = [10, 11]      # 牛頭 3+5 = 8、2枚
    env.rows[0, 3, :2] = [20, 21]      # 牛頭 3+1 = 4、2枚
    env.row_len[0] = [2, 1, 2, 2]
    env.pending_games = np.array([0], dtype=np.intp)
    env.pending_players = np.array([0], dtype=np.intp)
    env.pending_cards = np.array([100], dtype=np.int16)
    # 行0 と 行1 が牛頭2 で同点 → 枚数が少ない行1
    assert min_bull_rows(env, np.arange(1)).tolist() == [1]


def test_bots_play_legal_full_deals():
    for bot in ALL_BOTS:
        env = VecNimmt(64, 4, seed=3)
        taken = play_all_seats(env, bot)
        assert env.turn == 10
        assert (env.hands == 0).all()
        assert (taken >= 0).all()


def test_strength_ordering_random_greedy_heuristic():
    """同じ配牌で4席とも同じbotに打たせ、平均失点を比べる。
    heuristic < greedy < random になること（値は測定して記録する）。"""
    rng = np.random.default_rng(99)
    decks = new_decks(3000, rng)
    means = {}
    for bot in ALL_BOTS:
        env = VecNimmt(3000, 4, seed=0)
        env.reset(decks)
        means[bot.name] = float(play_all_seats(env, bot).mean())
    print(means)
    assert means["heuristic"] < means["greedy"] < means["random"]
