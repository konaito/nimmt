"""可変長の意思決定系列を貯めて GAE を計算するバッファ。

キーは (ゲーム, プレイヤー) の組。1つの「波」で1キーにつき最大1件しか追加されないので、
step カウンタと (キー, step) → レコード index の行列だけで系列を復元できる。
"""

from dataclasses import dataclass

import numpy as np

from ..cards import HAND_SIZE, N_ROWS
from ..features import C_DIM, G_DIM, R_DIM

CARD, ROW = 0, 1


@dataclass
class Batch:
    card_g: np.ndarray
    card_x: np.ndarray
    card_mask: np.ndarray
    card_a: np.ndarray
    card_logp: np.ndarray
    card_adv: np.ndarray
    card_ret: np.ndarray
    row_g: np.ndarray
    row_x: np.ndarray
    row_a: np.ndarray
    row_logp: np.ndarray
    row_adv: np.ndarray
    row_ret: np.ndarray
    mean_return: float


class TrajectoryBuffer:
    def __init__(self, n_keys: int, max_steps: int = 20):
        self.n_keys = int(n_keys)
        self.max_steps = int(max_steps)
        cap = self.n_keys * self.max_steps
        self.rec_key = np.zeros(cap, np.int64)
        self.rec_step = np.zeros(cap, np.int32)
        self.rec_kind = np.zeros(cap, np.int8)
        self.rec_local = np.zeros(cap, np.int64)
        self.rec_value = np.zeros(cap, np.float32)
        self.rec_logp = np.zeros(cap, np.float32)
        self.rec_reward = np.zeros(cap, np.float32)
        self._n = 0
        self.step = np.zeros(self.n_keys, np.int32)
        self.last = np.full(self.n_keys, -1, np.int64)
        self._card = {"g": [], "x": [], "mask": [], "a": []}
        self._row = {"g": [], "x": [], "a": []}
        self._n_card = 0
        self._n_row = 0

    def _add(self, keys, kind, local, logp, value):
        k = np.asarray(keys, np.int64)
        n = k.size
        idx = self._n + np.arange(n, dtype=np.int64)
        assert self._n + n <= self.rec_key.size, "max_steps を超えた"
        self.rec_key[idx] = k
        self.rec_step[idx] = self.step[k]
        self.rec_kind[idx] = kind
        self.rec_local[idx] = local
        self.rec_logp[idx] = logp
        self.rec_value[idx] = value
        self.rec_reward[idx] = 0.0
        self.step[k] += 1
        self.last[k] = idx
        self._n += n

    def add_card(self, keys, g, cards, mask, action, logp, value):
        n = np.asarray(keys).size
        local = self._n_card + np.arange(n, dtype=np.int64)
        self._card["g"].append(np.asarray(g, np.float32))
        self._card["x"].append(np.asarray(cards, np.float32))
        self._card["mask"].append(np.asarray(mask, bool))
        self._card["a"].append(np.asarray(action, np.int64))
        self._n_card += n
        self._add(keys, CARD, local, logp, value)

    def add_row(self, keys, g, rows, action, logp, value):
        n = np.asarray(keys).size
        local = self._n_row + np.arange(n, dtype=np.int64)
        self._row["g"].append(np.asarray(g, np.float32))
        self._row["x"].append(np.asarray(rows, np.float32))
        self._row["a"].append(np.asarray(action, np.int64))
        self._n_row += n
        self._add(keys, ROW, local, logp, value)

    def add_reward(self, keys, rewards):
        k = np.asarray(keys, np.int64)
        idx = self.last[k]
        assert (idx >= 0).all(), "報酬より先に transition が必要"
        np.add.at(self.rec_reward, idx, np.asarray(rewards, np.float32))

    def finish(self, gamma: float = 1.0, lam: float = 0.95, normalize: bool = True) -> Batch:
        n, K, T = self._n, self.n_keys, self.max_steps
        idx_mat = np.full((K, T), -1, np.int64)
        idx_mat[self.rec_key[:n], self.rec_step[:n]] = np.arange(n, dtype=np.int64)
        valid = idx_mat >= 0
        safe = np.maximum(idx_mat, 0)

        V = np.where(valid, self.rec_value[safe], 0.0).astype(np.float32)
        R = np.where(valid, self.rec_reward[safe], 0.0).astype(np.float32)

        adv_mat = np.zeros((K, T), np.float32)
        running = np.zeros(K, np.float32)
        next_v = np.zeros(K, np.float32)
        for t in range(T - 1, -1, -1):
            m = valid[:, t]
            delta = R[:, t] + gamma * next_v - V[:, t]
            running = np.where(m, delta + gamma * lam * running, running)
            adv_mat[:, t] = np.where(m, running, 0.0)
            next_v = np.where(m, V[:, t], next_v)

        adv = np.zeros(n, np.float32)
        adv[idx_mat[valid]] = adv_mat[valid]
        ret = adv + self.rec_value[:n]
        first = ret[self.rec_step[:n] == 0]
        mean_return = float(first.mean()) if first.size else 0.0
        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        kind = self.rec_kind[:n]
        loc = self.rec_local[:n]
        c_sel = kind == CARD
        r_sel = kind == ROW
        c_ord = np.argsort(loc[c_sel])
        r_ord = np.argsort(loc[r_sel])

        def cat(chunks, dtype, tail=()):
            """片相の経験が0件でも (0, *tail) の形を保つ。Batch の shape 契約のため。"""
            return (np.concatenate(chunks, axis=0) if chunks
                    else np.zeros((0, *tail), dtype=dtype))

        return Batch(
            card_g=cat(self._card["g"], np.float32, (G_DIM,)),
            card_x=cat(self._card["x"], np.float32, (HAND_SIZE, C_DIM)),
            card_mask=cat(self._card["mask"], bool, (HAND_SIZE,)),
            card_a=cat(self._card["a"], np.int64),
            card_logp=self.rec_logp[:n][c_sel][c_ord],
            card_adv=adv[c_sel][c_ord],
            card_ret=ret[c_sel][c_ord],
            row_g=cat(self._row["g"], np.float32, (G_DIM,)),
            row_x=cat(self._row["x"], np.float32, (N_ROWS, R_DIM)),
            row_a=cat(self._row["a"], np.int64),
            row_logp=self.rec_logp[:n][r_sel][r_ord],
            row_adv=adv[r_sel][r_ord],
            row_ret=ret[r_sel][r_ord],
            mean_return=mean_return,
        )
