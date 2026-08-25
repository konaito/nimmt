import numpy as np

from nimmt.arena import paired_ci, run_match, summarize
from nimmt.bots.rollout import MCRolloutBot
from nimmt.bots.simple import HeuristicBot
from nimmt.vec import PHASE_ROW, VecNimmt, new_decks


def test_from_state_round_trips():
    a = VecNimmt(4, 4, seed=1)
    b = VecNimmt.from_state(a.rows, a.row_len, a.hands, a.taken, a.seen, a.turn)
    assert (b.rows == a.rows).all() and (b.hands == a.hands).all()
    assert (b.row_len == a.row_len).all() and (b.seen == a.seen).all()
    assert b.turn == a.turn and b.B == a.B and b.P == a.P
    b.rows[0, 0, 0] = 99
    assert a.rows[0, 0, 0] != 99, "コピーされていない"


def test_rollout_bot_plays_legal_moves():
    env = VecNimmt(16, 4, seed=2)
    bot = MCRolloutBot(n_samples=4, top_k=3, seed=0)
    games = np.repeat(np.arange(16), 4)
    players = np.tile(np.arange(4), 16)
    while not env.done():
        slots = bot.pick_cards(env, games, players).reshape(16, 4)
        assert (env.hands[games, players, slots.ravel()] > 0).all()
        env.step_cards(slots)
        while env.phase == PHASE_ROW:
            env.step_rows(bot.pick_rows(env, np.arange(env.pending_games.size)))
    assert env.turn == 10


def test_rollout_beats_heuristic():
    """4人戦、mc-rollout 2席 vs heuristic 2席。重い（数分）のでディール数は控えめ。"""
    decks = new_decks(400, np.random.default_rng(21))
    bots = [MCRolloutBot(n_samples=8, top_k=4, seed=0), HeuristicBot(),
            MCRolloutBot(n_samples=8, top_k=4, seed=1), HeuristicBot()]
    per_deal = run_match(bots, decks, batch=400)
    r = summarize(per_deal, [b.name + str(i) for i, b in enumerate(bots)])
    print(dict(zip(r.names, r.mean_loss.round(3).tolist())))
    mc = (per_deal[:, 0] + per_deal[:, 2]) / 2.0
    he = (per_deal[:, 1] + per_deal[:, 3]) / 2.0
    diff = float(mc.mean() - he.mean())
    print(f"mc - heuristic = {diff:.3f}")
    assert diff < 0
