"""Bot の共通インターフェース。

pick_cards / pick_rows は「任意の (ゲーム, プレイヤー) の組」に対して手を返す。
arena はこれを使って席ごとに別の bot を割り当てる。
"""

from abc import ABC, abstractmethod

import numpy as np

_INF = np.float32(1e9)


class Bot(ABC):
    name: str = "bot"

    @abstractmethod
    def pick_cards(self, env, games: np.ndarray, players: np.ndarray) -> np.ndarray:
        """(M,) 手札スロット index を返す。"""

    @abstractmethod
    def pick_rows(self, env, sel: np.ndarray) -> np.ndarray:
        """sel は env.pending_* への index。(K,) の行 index を返す。"""


def argmin_masked(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(M,10) のコストを mask 内で argmin。無効スロットは選ばれない。"""
    return np.where(mask, cost, _INF).argmin(axis=1)


def min_bull_rows(env, sel: np.ndarray) -> np.ndarray:
    """牛頭最小の行。同点なら枚数が少ない方、さらに同点なら index が小さい方。"""
    games = env.pending_games[sel]
    bulls = env.row_points_all(games).astype(np.float32)
    lens = env.row_len[games].astype(np.float32)
    return (bulls * 10.0 + lens).argmin(axis=1)
