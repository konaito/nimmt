"""Bradley-Terry モデルの Elo レーティング推定。

内部は Newton-Raphson（対数尤度の勾配とヘッシアンから毎回 n×n の線形方程式を解く）。
素朴な MM 反復は同じ不動点に収束するが収束が線形で非常に遅く、100勝0敗規模のデータでは
iters=100 で真値から 21 Elo ずれる（実測）。Newton は同じケースで10反復未満で機械精度に達する。
ヘッシアンは対戦グラフの重み付きラプラシアンで半正定値（零空間は定数ベクトル＝並進不変の自由度）
なので、`np.linalg.lstsq` の最小ノルム解を取ってその自由度を吸収する。
"""

import numpy as np

_MAX_BACKTRACK = 40


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """オーバーフローしない σ(x)。x が大きく負のとき exp(-x) が inf になるのを避ける。"""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _log_likelihood(w: np.ndarray, s: np.ndarray) -> float:
    """Bradley-Terry の対数尤度 Σ w_ij log σ(s_i - s_j)。logaddexp で安定に計算する。"""
    diff = s[:, None] - s[None, :]
    return float((w * -np.logaddexp(0.0, -diff)).sum())


def bradley_terry(wins: np.ndarray, iters: int = 500, tol: float = 1e-10,
                  prior: float = 0.5) -> np.ndarray:
    """wins[i,j] = i の j に対する勝ち数。返り値は平均0の Elo レーティング。

    無敗（または全敗）のプレイヤーがいると Bradley-Terry の最尤推定は存在せず、
    素朴な反復は収束しないまま `iters` の打ち切り位置で決まる恣意的な値を静かに返す。
    実際に対戦したペアにだけ `prior` 分の擬似勝ちを両側へ足して、必ず有限解にする。
    ペアあたり数千ディール回すので 0.5 の擬似対戦の影響は無視できる。

    誰とも対戦していないプレイヤーはレーティングを定義できないので、
    黙って無意味な値を返さずに ValueError にする。
    """
    w = np.asarray(wins, dtype=np.float64)
    n = w.shape[0]
    played = (w + w.T) > 0
    if n > 1 and not played.any(axis=1).all():
        raise ValueError("誰とも対戦していないプレイヤーがいる（Elo を定義できない）")
    w = w + prior * played
    games = w + w.T
    total_wins = w.sum(axis=1)

    s = np.zeros(n, dtype=np.float64)
    ll = _log_likelihood(w, s)
    for _ in range(iters):
        diff = s[:, None] - s[None, :]
        sig = _sigmoid(diff)
        grad = total_wins - (games * sig).sum(axis=1)
        weight = games * sig * (1.0 - sig)
        hess_neg = -weight.copy()
        np.fill_diagonal(hess_neg, weight.sum(axis=1))
        delta, *_ = np.linalg.lstsq(hess_neg, grad, rcond=None)

        # バックトラッキング直線探索。素の Newton は極端に偏った疎なデータで行き過ぎ、
        # σ が 0/1 に飽和してヘッシアンが0になる。すると delta=0 になって tol 判定が発火し、
        # 「収束した」ことにしてゴミを返す（実測で max|Elo| が 3e7 に飛ぶ）。
        # 対数尤度が上がる歩幅まで半分にしていくことでこれを防ぐ。
        step = 1.0
        improved = False
        for _b in range(_MAX_BACKTRACK):
            s_new = s + step * delta
            s_new -= s_new.mean()
            ll_new = _log_likelihood(w, s_new)
            if ll_new >= ll:
                improved = True
                break
            step *= 0.5
        if not improved:
            break

        converged = np.max(np.abs(s_new - s)) < tol
        s, ll = s_new, ll_new
        if converged:
            break

    elo = 400.0 * s / np.log(10.0)
    return elo - elo.mean()
