import numpy as np
import torch

from nimmt.features import C_DIM, G_DIM, R_DIM
from nimmt.rl.buffer import TrajectoryBuffer
from nimmt.rl.model import NimmtNet
from nimmt.rl.ppo import PPOConfig, ppo_update
from nimmt.rl.rollout import REWARD_SCALE, collect_selfplay
from nimmt.vec import VecNimmt


def test_gae_with_gamma_one_gives_return_to_go():
    """γ=1, λ=1, value=0 なら advantage は残り報酬の合計に一致する。"""
    buf = TrajectoryBuffer(n_keys=1, max_steps=4)
    for r in (-1.0, -2.0, -3.0):
        buf.add_card(
            np.array([0]), np.zeros((1, G_DIM), np.float32),
            np.zeros((1, 10, C_DIM), np.float32), np.ones((1, 10), bool),
            np.array([0]), np.zeros(1, np.float32), np.zeros(1, np.float32),
        )
        buf.add_reward(np.array([0]), np.array([r], np.float32))
    b = buf.finish(gamma=1.0, lam=1.0, normalize=False)
    assert np.allclose(b.card_adv, [-6.0, -5.0, -3.0])
    assert np.allclose(b.card_ret, [-6.0, -5.0, -3.0])


def test_reward_attaches_to_last_transition_of_the_turn():
    """カード選択のあとに行選択がある場合、報酬は行選択に付く。"""
    buf = TrajectoryBuffer(n_keys=1, max_steps=4)
    buf.add_card(np.array([0]), np.zeros((1, G_DIM), np.float32),
                 np.zeros((1, 10, C_DIM), np.float32), np.ones((1, 10), bool),
                 np.array([0]), np.zeros(1, np.float32), np.zeros(1, np.float32))
    buf.add_row(np.array([0]), np.zeros((1, G_DIM), np.float32),
                np.zeros((1, 4, R_DIM), np.float32),
                np.array([2]), np.zeros(1, np.float32), np.zeros(1, np.float32))
    buf.add_reward(np.array([0]), np.array([-5.0], np.float32))
    b = buf.finish(gamma=1.0, lam=1.0, normalize=False)
    assert np.allclose(b.card_adv, [-5.0])
    assert np.allclose(b.row_adv, [-5.0])
    assert b.row_a.tolist() == [2]


def test_collect_selfplay_shapes_and_return_consistency():
    net = NimmtNet(hidden=32, depth=2)
    env = VecNimmt(64, 4, seed=8)
    buf = TrajectoryBuffer(n_keys=64 * 4)
    taken = collect_selfplay(net, env, buf, temperature=1.0, rng=np.random.default_rng(0))
    assert taken.shape == (64, 4)
    b = buf.finish(lam=1.0, normalize=False)   # λ=1 のときだけ ret は return-to-go に一致する
    # カード意思決定はちょうど 64*4*10 回
    assert b.card_g.shape == (64 * 4 * 10, G_DIM)
    assert b.card_x.shape == (64 * 4 * 10, 10, C_DIM)
    assert b.row_g.shape[0] == b.row_x.shape[0] == b.row_a.shape[0]
    assert b.row_x.shape[1:] == (4, R_DIM)
    # 各キーの最初の transition のリターンは、そのディールの総失点の負値/スケール
    first_ret = b.card_ret[: 64 * 4]   # wave 0 はキー順に並んでいる
    assert np.allclose(first_ret, -taken.ravel() / REWARD_SCALE, atol=1e-4)


def test_collect_league_records_only_the_learner_and_offsets_keys():
    """リーグ収集は学習者席の経験だけを記録し、key_offset でキーが衝突しないこと。

    key_offset がずれると「学習は回るが経験が壊れている」形で静かに死ぬので、
    キー空間そのものを直接検査する。
    """
    from nimmt.rl.rollout import collect_league

    for learner_seat, key_offset, B, P in ((0, 0, 8, 4), (2, 5, 8, 4)):
        net = NimmtNet(hidden=16, depth=1)
        opp = NimmtNet(hidden=16, depth=1)
        env = VecNimmt(B, P, seed=11)
        buf = TrajectoryBuffer(n_keys=(B + key_offset) * P)
        taken = collect_league(
            net, opp, env, buf, learner_seat=learner_seat,
            rng=np.random.default_rng(2), key_offset=key_offset,
        )
        b = buf.finish(lam=1.0, normalize=False)

        # 学習者席のカード決定はちょうど B*10 回。他席の経験は1件も入っていない
        assert b.card_g.shape[0] == B * 10
        # 使われたキーが学習者キーの集合と完全一致（= key_offset の衝突回避の直接証明）
        used = np.unique(buf.rec_key[: buf._n])
        expected = (np.arange(B) + key_offset) * P + learner_seat
        assert np.array_equal(used, expected)
        assert used.max() < buf.n_keys
        # 最初の transition のリターンが学習者席の最終失点と一致
        assert np.allclose(b.card_ret[:B], -taken[:, learner_seat] / REWARD_SCALE, atol=1e-4)


def test_ppo_update_produces_finite_metrics():
    torch.manual_seed(0)
    net = NimmtNet(hidden=32, depth=2)
    env = VecNimmt(64, 4, seed=9)
    buf = TrajectoryBuffer(n_keys=64 * 4)
    collect_selfplay(net, env, buf, temperature=1.0, rng=np.random.default_rng(1))
    batch = buf.finish()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    cfg = PPOConfig(minibatch=1024, epochs=2)
    m = ppo_update(net, opt, batch, cfg)
    for k in ("policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl"):
        assert k in m and np.isfinite(m[k])
    assert m["entropy"] > 0


def test_learning_signal_beats_random_after_short_training():
    """40イテレーションだけ回して、random より失点が下がることを確認する（スモークテスト）。"""
    from nimmt.arena import run_match, summarize
    from nimmt.bots.neural import NeuralBot
    from nimmt.bots.simple import RandomBot
    from nimmt.vec import new_decks

    torch.manual_seed(0)
    net = NimmtNet(hidden=64, depth=2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    cfg = PPOConfig(minibatch=4096, epochs=2)
    rng = np.random.default_rng(0)
    for _ in range(40):
        env = VecNimmt(256, 4, seed=int(rng.integers(1 << 30)))
        buf = TrajectoryBuffer(n_keys=256 * 4)
        collect_selfplay(net, env, buf, temperature=1.0, rng=rng)
        ppo_update(net, opt, buf.finish(), cfg)

    decks = new_decks(600, np.random.default_rng(77))
    per_deal = run_match([NeuralBot(net), RandomBot(seed=0)], decks)
    r = summarize(per_deal, ["neural", "random"])
    print(dict(zip(r.names, r.mean_loss.round(3).tolist())))
    assert r.mean_loss[0] < r.mean_loss[1]
