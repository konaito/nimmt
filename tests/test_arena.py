import numpy as np
import pytest

from nimmt.arena import (MatchResult, paired_ci, paired_ci_field, pairwise_wins,
                         run_match, summarize)
from nimmt.bots.simple import GreedyBot, HeuristicBot, RandomBot
from nimmt.elo import bradley_terry
from nimmt.vec import new_decks


def test_identical_deterministic_bots_are_symmetric():
    """同じ決定的botを2席に置いて席ローテーションすれば、失点は完全に一致するはず。"""
    decks = new_decks(200, np.random.default_rng(1))
    per_deal = run_match([GreedyBot(), GreedyBot()], decks)
    assert per_deal.shape == (400, 2)
    assert per_deal[:, 0].sum() == per_deal[:, 1].sum()


def test_summarize_win_rates_sum_to_one():
    per_deal = np.array([[3, 5], [7, 7], [1, 0]], dtype=np.int16)
    r = summarize(per_deal, ["a", "b"])
    assert isinstance(r, MatchResult)
    assert r.mean_loss.tolist() == [11 / 3, 12 / 3]
    # deal0: a勝ち, deal1: 引き分け(0.5ずつ), deal2: b勝ち
    assert r.win_rate.tolist() == [(1 + 0.5) / 3, (0.5 + 1) / 3]
    assert abs(r.win_rate.sum() - 1.0) < 1e-9


def test_pairwise_wins_counts_ties_as_half():
    per_deal = np.array([[3, 5], [7, 7], [1, 0]], dtype=np.int16)
    w = pairwise_wins(per_deal)
    assert w.shape == (2, 2)
    assert w[0, 1] == 1.5 and w[1, 0] == 1.5


def test_paired_ci_detects_real_difference():
    decks = new_decks(2000, np.random.default_rng(11))
    per_deal = run_match([HeuristicBot(), RandomBot()], decks)
    diff, lo, hi = paired_ci(per_deal, 0, 1, n_boot=2000, seed=0)
    print(f"heuristic - random = {diff:.3f} [{lo:.3f}, {hi:.3f}]")
    assert diff < 0            # heuristic の方が失点が少ない
    assert hi < 0              # 95%CI が 0 を跨がない


def test_paired_ci_field_matches_paired_ci_for_two_players():
    """2人戦では「残り全席の平均」は相手1人そのものなので、両者は一致する。"""
    per_deal = np.array([[3, 5], [7, 7], [1, 0], [9, 2]], dtype=np.int16)
    a = paired_ci(per_deal, 0, 1, n_boot=500, seed=0)
    b = paired_ci_field(per_deal, 0, n_boot=500, seed=0)
    assert a == b


def test_paired_ci_field_uses_mean_of_all_others():
    per_deal = np.array([[0, 4, 8], [2, 2, 2]], dtype=np.int16)
    diff, _, _ = paired_ci_field(per_deal, 0, n_boot=100, seed=0)
    # deal0: 0 - 6 = -6, deal1: 2 - 2 = 0 → 平均 -3
    assert abs(diff - (-3.0)) < 1e-9


def test_paired_ci_on_identical_bots_includes_zero():
    decks = new_decks(500, np.random.default_rng(12))
    per_deal = run_match([GreedyBot(), GreedyBot()], decks)
    diff, lo, hi = paired_ci(per_deal, 0, 1, n_boot=1000, seed=0)
    assert lo <= 0 <= hi


def test_four_player_ordering_and_elo():
    """4人戦。heuristic < greedy < random（失点が少ないほど強い）。Eloも同じ順序。"""
    decks = new_decks(1500, np.random.default_rng(13))
    bots = [HeuristicBot(), GreedyBot(), RandomBot(seed=1), RandomBot(seed=2)]
    per_deal = run_match(bots, decks)
    r = summarize(per_deal, [b.name for b in bots])
    print(dict(zip(r.names, r.mean_loss.round(3).tolist())))
    assert r.mean_loss[0] < r.mean_loss[1] < r.mean_loss[2]

    ratings = bradley_terry(pairwise_wins(per_deal))
    print(ratings.round(1).tolist())
    assert ratings[0] > ratings[1] > ratings[2]
    assert abs(ratings.mean()) < 1e-6


def test_bradley_terry_recovers_known_ordering():
    wins = np.array(
        [[0, 80, 95], [20, 0, 70], [5, 30, 0]], dtype=float
    )
    r = bradley_terry(wins)
    assert r[0] > r[1] > r[2]


def test_bradley_terry_is_stable_with_an_undefeated_player():
    """無敗のプレイヤーがいても MLE が存在し、iters に依存しない有限値になること。
    擬似対戦の prior を入れないと iters に比例して発散する。"""
    wins = np.array([[0, 100, 100], [0, 0, 50], [0, 50, 0]], dtype=float)
    r_short = bradley_terry(wins, iters=100)
    r_long = bradley_terry(wins, iters=10000)
    assert np.all(np.isfinite(r_short))
    assert np.allclose(r_short, r_long, atol=1e-6)
    assert r_short[0] > r_short[1] and r_short[0] > r_short[2]


def test_bradley_terry_rejects_isolated_player():
    """誰とも対戦していないプレイヤーは黙って無意味な値を返さず ValueError にする。"""
    wins = np.array([[0, 10, 0], [10, 0, 0], [0, 0, 0]], dtype=float)
    with pytest.raises(ValueError):
        bradley_terry(wins)


def test_bradley_terry_does_not_diverge_on_lopsided_sparse_data():
    """極端に偏った疎な対戦成績でも真の最尤推定に収束すること。

    バックトラッキング直線探索が無い素の Newton はこの入力で発散し、σ が飽和して
    ヘッシアンが0になるため「収束した」ことにしてゴミを返す
    （実測: max|Elo| = 3.2e7、勾配 39 のまま停止、対数尤度 -1.03e7）。
    減衰を入れると 17 反復で対数尤度 -19953.37、勾配 2.9e-11 に到達する。
    """
    from nimmt.elo import _log_likelihood

    wins = np.array(
        [[0, 0, 247499, 959, 27],
         [0, 0, 0, 0, 38],
         [2580, 9481, 0, 0, 0],
         [23, 0, 0, 0, 909],
         [8, 0, 0, 90254, 0]],
        dtype=float,
    )
    r = bradley_terry(wins)
    assert np.all(np.isfinite(r))
    assert np.max(np.abs(r)) < 5000.0
    # 返り値から内部の強さパラメータを復元して対数尤度を見る（凹関数なので最尤に近いほど大きい）
    s = r * np.log(10.0) / 400.0
    w = wins + 0.5 * ((wins + wins.T) > 0)
    assert _log_likelihood(w, s) > -20100.0


def test_boot_ci_is_independent_of_chunk_size(monkeypatch):
    """メモリ節約のチャンク分割が結果を変えないこと（乱数ストリームの消費順は同じ）。"""
    import nimmt.arena as arena

    per_deal = np.array([[3, 5], [7, 7], [1, 0], [9, 2], [4, 6]], dtype=np.int16)
    monkeypatch.setattr(arena, "_BOOT_MAX_CELLS", 8_000_000)
    big = arena.paired_ci(per_deal, 0, 1, n_boot=1000, seed=3)
    monkeypatch.setattr(arena, "_BOOT_MAX_CELLS", 10)
    small = arena.paired_ci(per_deal, 0, 1, n_boot=1000, seed=3)
    assert big == small
