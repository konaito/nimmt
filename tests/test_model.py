import numpy as np
import torch

from nimmt.bots.neural import NeuralBot
from nimmt.features import C_DIM, G_DIM, R_DIM
from nimmt.rl.model import NimmtNet, load, save
from nimmt.vec import PHASE_ROW, VecNimmt


def test_shapes_and_masking():
    net = NimmtNet(hidden=32, depth=2)
    g = torch.randn(5, G_DIM)
    cards = torch.randn(5, 10, C_DIM)
    mask = torch.zeros(5, 10, dtype=torch.bool)
    mask[:, :3] = True
    h = net.encode(g)
    assert h.shape == (5, 32)
    lg = net.card_logits(h, cards, mask)
    assert lg.shape == (5, 10)
    assert torch.isinf(lg[:, 3:]).all() and (lg[:, 3:] < 0).all()
    p = torch.softmax(lg, dim=1)
    assert torch.allclose(p[:, 3:], torch.zeros_like(p[:, 3:]))
    assert torch.allclose(p.sum(1), torch.ones(5))
    assert net.row_logits(h, torch.randn(5, 4, R_DIM)).shape == (5, 4)
    assert net.value(h).shape == (5,)


def test_save_load_round_trip(tmp_path):
    net = NimmtNet(hidden=48, depth=3)
    p = tmp_path / "net.pt"
    save(net, p)
    net2 = load(p)
    assert net2.hidden == 48 and net2.depth == 3
    g = torch.randn(3, G_DIM)
    assert torch.allclose(net.encode(g), net2.encode(g), atol=1e-6)


def test_neural_bot_plays_legal_full_deal():
    net = NimmtNet(hidden=32, depth=2)
    bot = NeuralBot(net, temperature=1.0, seed=0)
    env = VecNimmt(32, 4, seed=4)
    games = np.repeat(np.arange(32), 4)
    players = np.tile(np.arange(4), 32)
    while not env.done():
        slots = bot.pick_cards(env, games, players).reshape(32, 4)
        assert (env.hands[games, players, slots.ravel()] > 0).all()
        env.step_cards(slots)
        while env.phase == PHASE_ROW:
            r = bot.pick_rows(env, np.arange(env.pending_games.size))
            assert ((r >= 0) & (r < 4)).all()
            env.step_rows(r)
    assert env.turn == 10


def test_sampling_never_picks_a_masked_slot():
    """マスクされたスロット（-inf）が選ばれないこと、かつ正しい分布でサンプルされること。

    累積和方式は float32 の丸めで最後のスロットに落ちる経路があった。
    """
    net = NimmtNet(hidden=32, depth=2)
    bot = NeuralBot(net, temperature=1.0, seed=0)
    logits = np.full((200000, 10), -np.inf, dtype=np.float32)
    logits[:, 2] = 0.0
    logits[:, 5] = 3.0
    a = bot._sample(logits)
    assert set(np.unique(a).tolist()) <= {2, 5}
    expected = float(np.exp(3.0) / (np.exp(3.0) + 1.0))
    assert abs(float((a == 5).mean()) - expected) < 0.01


def test_greedy_bot_is_deterministic():
    net = NimmtNet(hidden=32, depth=2)
    bot = NeuralBot(net, temperature=0.0)
    env = VecNimmt(8, 4, seed=6)
    games = np.repeat(np.arange(8), 4)
    players = np.tile(np.arange(4), 8)
    a = bot.pick_cards(env, games, players)
    b = bot.pick_cards(env, games, players)
    assert (a == b).all()
