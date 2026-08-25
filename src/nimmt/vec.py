"""ベクトル化ゲーム環境。B ゲーム × P 人を同時に進める。

意思決定は二相。カード選択のあと、行選択が必要なゲームだけを phase=PHASE_ROW で問い直す。
1ターンの解決は「公開された P 枚を昇順に処理」だが、各 k のステップで B ゲーム全部を
一括処理できるので、Python 側のループは最大 P 回で済む。
"""

import numpy as np

from .cards import BULL, HAND_SIZE, N_CARDS, N_ROWS, ROW_CAP

PHASE_CARD = 0
PHASE_ROW = 1

_BIG = np.int16(32767)


def new_decks(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 104) int16。各行が 1..104 の並べ替え。"""
    return (np.argsort(rng.random((n, N_CARDS)), axis=1) + 1).astype(np.int16)


class VecNimmt:
    def __init__(self, n_games: int, n_players: int, seed: int = 0):
        assert 1 <= n_players <= 10   # 1 はテストで局面を組むためだけに許す
        assert n_players * HAND_SIZE + N_ROWS <= N_CARDS
        self.B = int(n_games)
        self.P = int(n_players)
        self.rng = np.random.default_rng(seed)
        self._bidx = np.arange(self.B, dtype=np.intp)
        self._pidx = np.arange(self.P, dtype=np.intp)
        self._ridx = np.arange(N_ROWS, dtype=np.intp)
        self._cap = np.arange(ROW_CAP)
        self.reset()

    # ---------- setup ----------

    @classmethod
    def from_state(cls, rows, row_len, hands, taken, seen, turn, seed: int = 0) -> "VecNimmt":
        """任意の局面から env を作る。ロールアウト用。配列はすべてコピーする。"""
        obj = cls.__new__(cls)
        obj.B, obj.P = int(hands.shape[0]), int(hands.shape[1])
        obj.rng = np.random.default_rng(seed)
        obj._bidx = np.arange(obj.B, dtype=np.intp)
        obj._pidx = np.arange(obj.P, dtype=np.intp)
        obj._ridx = np.arange(N_ROWS, dtype=np.intp)
        obj._cap = np.arange(ROW_CAP)
        obj.rows = np.array(rows, dtype=np.int16, copy=True)
        obj.row_len = np.array(row_len, dtype=np.int8, copy=True)
        obj.hands = np.array(hands, dtype=np.int16, copy=True)
        obj.taken = np.array(taken, dtype=np.int16, copy=True)
        obj.seen = np.array(seen, dtype=bool, copy=True)
        obj.turn = int(turn)
        obj.phase = PHASE_CARD
        obj._turn_pts = np.zeros((obj.B, obj.P), dtype=np.int16)
        obj._played = np.zeros((obj.B, obj.P), dtype=np.int16)
        obj._order = np.zeros((obj.B, obj.P), dtype=np.intp)
        obj._k = 0
        obj.pending_games = None
        obj.pending_players = None
        obj.pending_cards = None
        return obj

    def reset(self, decks: np.ndarray | None = None) -> None:
        B, P = self.B, self.P
        if decks is None:
            decks = new_decks(B, self.rng)
        decks = np.asarray(decks, dtype=np.int16)
        assert decks.shape == (B, N_CARDS)

        hands = decks[:, : P * HAND_SIZE].reshape(B, P, HAND_SIZE)
        self.hands = np.sort(hands, axis=2).astype(np.int16)
        self.rows = np.zeros((B, N_ROWS, ROW_CAP), dtype=np.int16)
        self.rows[:, :, 0] = decks[:, P * HAND_SIZE : P * HAND_SIZE + N_ROWS]
        self.row_len = np.ones((B, N_ROWS), dtype=np.int8)
        self.taken = np.zeros((B, P), dtype=np.int16)
        self.seen = np.zeros((B, N_CARDS + 1), dtype=bool)
        np.put_along_axis(self.seen, self.rows[:, :, 0].astype(np.intp), True, axis=1)

        self.turn = 0
        self.phase = PHASE_CARD
        self._turn_pts = np.zeros((B, P), dtype=np.int16)
        self._played = np.zeros((B, P), dtype=np.int16)
        self._order = np.zeros((B, P), dtype=np.intp)
        self._k = 0
        self.pending_games = None
        self.pending_players = None
        self.pending_cards = None

    # ---------- queries ----------

    def done(self) -> bool:
        return self.turn >= HAND_SIZE

    def legal_mask(self) -> np.ndarray:
        return self.hands > 0

    def row_ends(self) -> np.ndarray:
        idx = (self.row_len - 1).astype(np.intp)
        return self.rows[self._bidx[:, None], self._ridx[None, :], idx]

    def row_points_all(self, games: np.ndarray) -> np.ndarray:
        """(M,4) — その行を今引き取ったときの牛頭。"""
        g = np.asarray(games, dtype=np.intp)
        rc = self.rows[g]                                        # (M,4,5)
        lens = self.row_len[g].astype(np.intp)                   # (M,4)
        mask = self._cap[None, None, :] < lens[:, :, None]
        return (BULL[rc] * mask).sum(axis=2).astype(np.int16)

    def last_turn_points(self) -> np.ndarray:
        return self._turn_pts.copy()

    # ---------- stepping ----------

    def step_cards(self, slots: np.ndarray) -> None:
        assert self.phase == PHASE_CARD, "phase が PHASE_CARD ではない"
        assert not self.done(), "ディールは既に終了している"
        s = np.asarray(slots, dtype=np.intp)
        assert s.shape == (self.B, self.P)

        cards = self.hands[self._bidx[:, None], self._pidx[None, :], s]
        assert (cards > 0).all(), "既に出したスロットが選ばれている"
        self.hands[self._bidx[:, None], self._pidx[None, :], s] = 0
        np.put_along_axis(self.seen, cards.astype(np.intp), True, axis=1)

        self._played = cards
        self._order = np.argsort(cards, axis=1, kind="stable").astype(np.intp)
        self._turn_pts[:] = 0
        self._k = 0
        self._advance()

    def step_rows(self, choices: np.ndarray) -> None:
        assert self.phase == PHASE_ROW, "phase が PHASE_ROW ではない"
        c = np.asarray(choices, dtype=np.intp)
        assert c.shape == self.pending_games.shape
        assert ((c >= 0) & (c < N_ROWS)).all()
        self._place(self.pending_games, self.pending_players, self.pending_cards, c, forced=True)
        self.pending_games = None
        self.pending_players = None
        self.pending_cards = None
        self.phase = PHASE_CARD
        self._k += 1
        self._advance()

    # ---------- internals ----------

    def _advance(self) -> None:
        while self._k < self.P:
            pl = self._order[:, self._k]                      # (B,)
            card = self._played[self._bidx, pl]               # (B,)
            ends = self.row_ends()                            # (B,4)
            valid = ends < card[:, None]
            any_valid = valid.any(axis=1)

            idx = np.flatnonzero(any_valid)
            if idx.size:
                scored = np.where(valid[idx], ends[idx], np.int16(-1))
                tgt = np.argmax(scored, axis=1).astype(np.intp)
                self._place(idx, pl[idx], card[idx], tgt, forced=False)

            need = np.flatnonzero(~any_valid)
            if need.size:
                self.pending_games = need
                self.pending_players = pl[need]
                self.pending_cards = card[need]
                self.phase = PHASE_ROW
                return
            self._k += 1

        self.turn += 1
        self.phase = PHASE_CARD

    def _place(self, g, p, card, tgt, forced: bool) -> None:
        """g の各ゲームで、プレイヤー p が card を行 tgt に置く。
        g はゲーム index の重複なしの配列（1つの k につきゲームごとに1人しか解決しない）。"""
        g = np.asarray(g, dtype=np.intp)
        p = np.asarray(p, dtype=np.intp)
        tgt = np.asarray(tgt, dtype=np.intp)
        card = np.asarray(card, dtype=np.int16)

        lens = self.row_len[g, tgt].astype(np.intp)
        take = np.ones(g.size, dtype=bool) if forced else (lens == ROW_CAP)

        rc = self.rows[g, tgt]                                   # (M,5)
        mask = self._cap[None, :] < lens[:, None]
        pts = (BULL[rc] * mask).sum(axis=1).astype(np.int16)

        tg, tp, tt, tc = g[take], p[take], tgt[take], card[take]
        if tg.size:
            self.taken[tg, tp] += pts[take]
            self._turn_pts[tg, tp] += pts[take]
            self.rows[tg, tt, :] = 0
            self.rows[tg, tt, 0] = tc
            self.row_len[tg, tt] = 1

        keep = ~take
        kg, kt, kl, kc = g[keep], tgt[keep], lens[keep], card[keep]
        if kg.size:
            self.rows[kg, kt, kl] = kc
            self.row_len[kg, kt] = (kl + 1).astype(np.int8)
