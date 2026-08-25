"""ベースライン bot 3種。評価の基準線であり、これ以上でなければ学習は失敗。"""

import numpy as np

from ..cards import BULL
from ..features import placement_info
from .base import Bot, argmin_masked, min_bull_rows


class RandomBot(Bot):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def pick_cards(self, env, games, players):
        mask = env.hands[np.asarray(games, np.intp), np.asarray(players, np.intp)] > 0
        noise = self.rng.random(mask.shape).astype(np.float32)
        return argmin_masked(noise, mask)

    def pick_rows(self, env, sel):
        return self.rng.integers(0, 4, size=np.asarray(sel).size)


class GreedyBot(Bot):
    """今このカードを単独で解決したときに引き取る牛頭が最小のカード。
    同点なら行末との差が小さいカード（強制取得は最後）。"""

    name = "greedy"

    def pick_cards(self, env, games, players):
        i = placement_info(env, games, players, need_unseen=False)
        gap = np.where(i["below_all"], 200.0, i["gap"])
        cost = i["take_now"] * 1000.0 + gap
        return argmin_masked(cost.astype(np.float32), i["mask"])

    def pick_rows(self, env, sel):
        return min_bull_rows(env, sel)


class HeuristicBot(Bot):
    """割り込まれ確率（行末と自分のカードの間にある未見カードの割合）で期待失点を見積もる。"""

    name = "heuristic"

    FORCED_PENALTY = 2.0
    ROW_LEN_WEIGHT = 0.6
    GAP_TIEBREAK = 1e-3

    def costs(self, env, games, players) -> np.ndarray:
        """(M,10) の期待失点コスト。MCRolloutBot も候補の絞り込みにこれを使う。"""
        i = placement_info(env, games, players)
        bull = BULL[i["cards"]].astype(np.float32)
        risk = i["in_gap_frac"]
        normal = risk * (i["t_bull"] + bull) + self.ROW_LEN_WEIGHT * i["t_len"] * risk
        cost = np.where(
            i["below_all"],
            np.broadcast_to(i["min_bull"][:, None], normal.shape) + self.FORCED_PENALTY,
            np.where(i["is_sixth"], i["t_bull"], normal),
        )
        return (cost + self.GAP_TIEBREAK * i["gap"]).astype(np.float32)

    def pick_cards(self, env, games, players):
        mask = env.hands[np.asarray(games, np.intp), np.asarray(players, np.intp)] > 0
        return argmin_masked(self.costs(env, games, players), mask)

    def pick_rows(self, env, sel):
        return min_bull_rows(env, sel)
