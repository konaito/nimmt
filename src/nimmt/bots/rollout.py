"""モンテカルロ・ロールアウト bot。未見カードから相手の手札をサンプリングし、
候補カードごとに残りディールをプレイアウトして期待失点が最小のものを選ぶ。評価専用。"""

import numpy as np

from ..cards import HAND_SIZE, N_CARDS
from ..vec import PHASE_ROW, VecNimmt
from .base import Bot, min_bull_rows
from .simple import HeuristicBot

_INF = np.float32(1e9)


class MCRolloutBot(Bot):
    name = "mc-rollout"

    def __init__(self, n_samples: int = 8, top_k: int = 4, playout: Bot | None = None,
                 chunk: int = 8192, seed: int = 0):
        self.n_samples = int(n_samples)
        self.top_k = int(top_k)
        self.playout = playout if playout is not None else HeuristicBot()
        self.chunk = int(chunk)
        self.rng = np.random.default_rng(seed)
        self._prior = HeuristicBot()

    # --- 候補の絞り込み: heuristic のコスト上位 top_k ---
    def _candidates(self, env, games, players):
        mask = env.hands[np.asarray(games, np.intp), np.asarray(players, np.intp)] > 0
        cost = np.where(mask, self._prior.costs(env, games, players), _INF)
        order = np.argsort(cost, axis=1)
        k = min(self.top_k, HAND_SIZE)
        cand = order[:, :k]                                   # (M,k)
        valid = np.take_along_axis(mask, cand, axis=1)
        return cand, valid, mask

    # --- 相手手札のサンプリング ---
    def _sample_hands(self, env, games, players, reps: int):
        """(M*reps, P, 10) の手札。自分の席は実際の手札、他席は未見からの無作為抽出。"""
        g = np.repeat(np.asarray(games, np.intp), reps)
        p = np.repeat(np.asarray(players, np.intp), reps)
        n = g.size
        remain = HAND_SIZE - env.turn

        avail = ~env.seen[g]                                   # (n,105)
        avail[:, 0] = False
        np.put_along_axis(avail, env.hands[g, p].astype(np.intp), False, axis=1)

        keys = self.rng.random((n, N_CARDS + 1)).astype(np.float32)
        keys[~avail] = 2.0
        order = np.argsort(keys, axis=1)                       # 未見カードが前に来る

        hands = np.zeros((n, env.P, HAND_SIZE), dtype=np.int16)
        cursor = 0
        for s in range(env.P):
            own = p == s
            # 圧縮(左詰め)しない: env.hands は既にプレイ済みスロットが0の (n,10) 形状。
            # 候補 index（_candidates が env.hands の元の並びに対して返す）と位置を
            # 一致させるため、詰め直さずそのままコピーする。
            hands[own, s, :] = env.hands[g[own], s, :]
            other = ~own
            take = order[other, cursor : cursor + remain].astype(np.int16)
            hands[other, s, :remain] = np.sort(take, axis=1)
            cursor += remain
        return g, p, hands

    def pick_cards(self, env, games, players):
        games = np.asarray(games, np.intp)
        players = np.asarray(players, np.intp)
        M = games.size
        cand, valid, mask = self._candidates(env, games, players)  # (M,k), (M,k), (M,10)
        k = cand.shape[1]
        reps = self.n_samples * k

        g, p, hands = self._sample_hands(env, games, players, reps)
        # (M, n_samples, k) の順に並んでいる前提で、各行がどの候補かを持たせる
        cand_ix = np.tile(np.arange(k), M * self.n_samples)
        # top_k > 残り手札枚数のとき cand は既に出したスロット(無効)を含みうる。
        # そのままプレイアウトに渡すと sub.step_cards の合法性チェックで落ちるので、
        # プレイアウト用には有効な代替スロット（この行の最初の合法スロット）に差し替える。
        # 最終選択は下で `valid` により無効候補を _INF で除外するので、
        # 差し替えても選ばれる結果には影響しない。
        fallback = mask.argmax(axis=1)                         # (M,) 必ず1枚以上合法な手がある
        sim_cand = np.where(valid, cand, fallback[:, None])     # (M,k)
        slot = sim_cand[np.repeat(np.arange(M), reps), cand_ix]

        loss = np.zeros(g.size, dtype=np.float32)
        for start in range(0, g.size, self.chunk):
            sl = slice(start, start + self.chunk)
            sub = VecNimmt.from_state(
                env.rows[g[sl]], env.row_len[g[sl]], hands[sl],
                np.zeros((g[sl].size, env.P), np.int16), env.seen[g[sl]], env.turn,
                seed=int(self.rng.integers(1 << 30)),
            )
            self._playout(sub, p[sl], slot[sl])
            loss[sl] = sub.taken[np.arange(sub.B), p[sl]].astype(np.float32)

        loss = loss.reshape(M, self.n_samples, k).mean(axis=1)   # (M,k)
        loss = np.where(valid, loss, _INF)
        best = loss.argmin(axis=1)
        return cand[np.arange(M), best]

    def _playout(self, sub, me, forced_slot):
        """最初の1手だけ forced_slot、それ以降は playout 方策で最後まで打つ。"""
        gidx = np.arange(sub.B, dtype=np.intp)
        first = True
        while not sub.done():
            games = np.repeat(gidx, sub.P)
            players = np.tile(np.arange(sub.P, dtype=np.intp), sub.B)
            slots = self.playout.pick_cards(sub, games, players).reshape(sub.B, sub.P)
            if first:
                slots[gidx, me] = forced_slot
                first = False
            sub.step_cards(slots)
            while sub.phase == PHASE_ROW:
                sub.step_rows(min_bull_rows(sub, np.arange(sub.pending_games.size)))

    def pick_rows(self, env, sel):
        return min_bull_rows(env, sel)
