"""重複配牌方式の対戦実行と統計。学習コードには依存しない。"""

from dataclasses import dataclass

import numpy as np

from .bots.base import Bot
from .vec import PHASE_ROW, VecNimmt


@dataclass
class MatchResult:
    names: list[str]
    mean_loss: np.ndarray
    win_rate: np.ndarray
    per_deal: np.ndarray


def _play_batch(env: VecNimmt, bots: list[Bot], seat_bot: list[int]) -> None:
    P = env.P
    games = np.arange(env.B, dtype=np.intp)
    while not env.done():
        slots = np.zeros((env.B, P), dtype=np.intp)
        for s in range(P):
            players = np.full(env.B, s, dtype=np.intp)
            slots[:, s] = bots[seat_bot[s]].pick_cards(env, games, players)
        env.step_cards(slots)
        while env.phase == PHASE_ROW:
            choices = np.zeros(env.pending_games.size, dtype=np.intp)
            for s in range(P):
                sel = np.flatnonzero(env.pending_players == s)
                if sel.size:
                    choices[sel] = bots[seat_bot[s]].pick_rows(env, sel)
            env.step_rows(choices)


def run_match(bots, decks, batch: int = 4096, rotations: int | None = None) -> np.ndarray:
    """同じ配牌セットを席ローテーションさせて対戦させる。(rotations*D, n_bots) の失点。

    再現性の注意: 配牌は `decks` で固定されるが、**乱数を持つ bot（RandomBot など）の結果は
    `batch` に依存する**。バッチ分割が `pick_cards` の呼び出し粒度を変え、bot 内部の
    Generator の消費順が変わるため。決定的な bot なら `batch` によらず結果は同一。
    同じ数字を再現したいときは `batch` を固定し、bot インスタンスを毎回作り直すこと
    （同一プロセスで `run_match` を2回呼ぶと乱数状態が引き継がれて別の結果になる）。

    `rotations` は既定で P（席数）。P の倍数でないと重複配牌による分散削減が効かない
    （不偏ではあるが分散が大きくなる）。
    """
    P = len(bots)
    R = P if rotations is None else int(rotations)
    decks = np.asarray(decks, dtype=np.int16)
    D = decks.shape[0]
    out = np.zeros((R, D, P), dtype=np.int16)

    for r in range(R):
        seat_bot = [(s + r) % P for s in range(P)]
        for start in range(0, D, batch):
            chunk = decks[start : start + batch]
            env = VecNimmt(chunk.shape[0], P, seed=0)
            env.reset(chunk)
            _play_batch(env, bots, seat_bot)
            for s in range(P):
                out[r, start : start + chunk.shape[0], seat_bot[s]] = env.taken[:, s]
    return out.reshape(R * D, P)


def summarize(per_deal: np.ndarray, names: list[str]) -> MatchResult:
    per_deal = np.asarray(per_deal)
    best = per_deal.min(axis=1, keepdims=True)
    is_best = per_deal == best
    share = is_best / is_best.sum(axis=1, keepdims=True)
    return MatchResult(
        names=list(names),
        mean_loss=per_deal.mean(axis=0).astype(np.float64),
        win_rate=share.mean(axis=0).astype(np.float64),
        per_deal=per_deal,
    )


# 一度に確保する index 行列の要素数の上限。int64 なので 8M 要素 = 約64MB。
# 分割しないと (n_boot × n_deals) を一気に確保する。n_boot=10000・8万ディールで 6.4GB になり、
# 16GB のマシンでは最終評価が落ちる。チャンクに切っても乱数ストリームの消費順は同じなので結果は不変。
_BOOT_MAX_CELLS = 8_000_000


def _boot_ci(d: np.ndarray, n_boot: int, seed: int):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    chunk = max(1, _BOOT_MAX_CELLS // max(n, 1))
    boots = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        boots[start : start + k] = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def paired_ci(per_deal, i: int, j: int, n_boot: int = 10000, seed: int = 0):
    """同一配牌での対応ありブートストラップ。(平均差, CI下限, CI上限)。負なら i が強い。"""
    x = np.asarray(per_deal).astype(np.float64)
    return _boot_ci(x[:, i] - x[:, j], n_boot, seed)


def paired_ci_field(per_deal, i: int, n_boot: int = 10000, seed: int = 0):
    """i と「残り全席の平均」の対応あり差。相手が複数席いるときはこちらを使う。
    席ごとに CI を出して境界を平均しても CI にはならない。"""
    x = np.asarray(per_deal).astype(np.float64)
    return _boot_ci(x[:, i] - np.delete(x, i, axis=1).mean(axis=1), n_boot, seed)


def pairwise_wins(per_deal: np.ndarray) -> np.ndarray:
    """wins[i,j] = i が j に勝った回数。引き分けは 0.5 ずつ。"""
    x = np.asarray(per_deal).astype(np.float64)
    n = x.shape[1]
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w[i, j] = (x[:, i] < x[:, j]).sum() + 0.5 * (x[:, i] == x[:, j]).sum()
    return w
