"""観測エンコーダ。グローバル特徴 / カードごと特徴 / 行ごと特徴 を作る。

`games` と `players` は同じ長さ M の index 配列で、任意の (ゲーム, プレイヤー) の組に対して
その視点の観測を作れる。カード相は全 (B,P) の組、行相は pending の組だけに使う。
"""

import numpy as np

from .cards import BULL, HAND_SIZE, N_CARDS, N_ROWS, ROW_CAP

G_DIM = 266
C_DIM = 12
R_DIM = 9

_MAX_OTHERS = 3          # 失点欄に載せる他プレイヤー数の上限
_BULL_SCALE = 14.0       # 行の牛頭のスケール（現実的な上限）
_TAKEN_SCALE = 20.0
_ONEHOT5 = np.eye(ROW_CAP, dtype=np.float32)
_ONEHOT_TURN = np.eye(HAND_SIZE, dtype=np.float32)
_ONEHOT_P = np.eye(9, dtype=np.float32)     # P = 2..10


def _hand_multihot(hands: np.ndarray) -> np.ndarray:
    """hands (M,10) int16 → (M,104) float32"""
    m = hands.shape[0]
    out = np.zeros((m, N_CARDS + 1), dtype=np.float32)
    np.put_along_axis(out, hands.astype(np.intp), 1.0, axis=1)
    out[:, 0] = 0.0
    return out[:, 1:]


def global_features(env, games, players, played_cards=None) -> np.ndarray:
    g = np.asarray(games, dtype=np.intp)
    p = np.asarray(players, dtype=np.intp)
    m = g.size
    out = np.zeros((m, G_DIM), dtype=np.float32)

    # --- 行 ---
    ends = env.row_ends()[g]                                   # (M,4)
    lens = env.row_len[g].astype(np.intp)                      # (M,4)
    bulls = env.row_points_all(g).astype(np.float32)           # (M,4)
    blk = np.concatenate(
        [
            (ends / N_CARDS).astype(np.float32)[:, :, None],
            (lens / ROW_CAP).astype(np.float32)[:, :, None],
            (bulls / _BULL_SCALE)[:, :, None],
            _ONEHOT5[lens - 1],
        ],
        axis=2,
    )                                                          # (M,4,8)
    out[:, 0:32] = blk.reshape(m, 32)

    # --- 手札 / 既知カード ---
    hands = env.hands[g, p]                                    # (M,10)
    out[:, 32:136] = _hand_multihot(hands)
    out[:, 136:240] = env.seen[g, 1:].astype(np.float32)

    # --- 失点 ---
    tk = env.taken[g].astype(np.float32)                       # (M,P)
    out[:, 240] = tk[np.arange(m), p] / _TAKEN_SCALE
    other_mask = np.arange(env.P)[None, :] != p[:, None]
    others = tk[other_mask].reshape(m, env.P - 1)
    others = -np.sort(-others, axis=1)[:, :_MAX_OTHERS]
    out[:, 241 : 241 + others.shape[1]] = others / _TAKEN_SCALE

    # --- ターン / 人数 ---
    out[:, 244:254] = _ONEHOT_TURN[min(env.turn, HAND_SIZE - 1)]
    out[:, 254] = env.P / 10.0
    out[:, 255:264] = _ONEHOT_P[max(env.P - 2, 0)]

    # --- 行相の追加情報 ---
    if played_cards is not None:
        pc = np.asarray(played_cards, dtype=np.intp)
        out[:, 264] = pc.astype(np.float32) / N_CARDS
        out[:, 265] = BULL[pc].astype(np.float32) / 7.0

    return out


def placement_info(env, games, players, need_unseen: bool = True) -> dict:
    """正規化前の生の配置情報。bot と特徴量の両方がこれを使う。

    `need_unseen=False` のとき、未見カードの累積和（`lower_frac`/`in_gap_frac`）を
    省いて零配列にする。返り値のキーと形は `need_unseen` の値によらず同一。
    """
    g = np.asarray(games, dtype=np.intp)
    p = np.asarray(players, dtype=np.intp)
    m = g.size

    hands = env.hands[g, p]                                    # (M,10)
    mask = hands > 0
    cards = hands.astype(np.int64)

    ends = env.row_ends()[g]
    lens = env.row_len[g].astype(np.int64)
    bulls = env.row_points_all(g).astype(np.float32)
    min_bull = bulls.min(axis=1)

    valid = ends[:, None, :] < hands[:, :, None]
    below_all = ~valid.any(axis=2)
    scored = np.where(valid, ends[:, None, :], np.int16(-1))
    tgt = np.argmax(scored, axis=2).astype(np.int64)

    rows_ix = np.arange(m)[:, None]
    t_end = ends[rows_ix, tgt].astype(np.int64)
    t_len = lens[rows_ix, tgt]
    t_bull = bulls[rows_ix, tgt]

    if need_unseen:
        unseen = (~env.seen[g]).astype(np.int32)
        np.put_along_axis(unseen, np.maximum(cards, 0), 0, axis=1)
        unseen[:, 0] = 0
        prefix = np.cumsum(unseen, axis=1)
        n_unseen = np.maximum(prefix[:, N_CARDS], 1).astype(np.float32)[:, None]

        lower = np.take_along_axis(prefix, np.maximum(cards - 1, 0), axis=1).astype(np.float32)
        at_end = np.take_along_axis(prefix, np.clip(t_end, 0, N_CARDS), axis=1).astype(np.float32)
        in_gap = np.where(below_all, 0.0, lower - at_end).astype(np.float32)
        lower_frac = (lower / n_unseen).astype(np.float32)
        in_gap_frac = (in_gap / n_unseen).astype(np.float32)
    else:
        # gap はこの分岐より後で計算されるので zeros_like は使えない。形を直に書く
        lower_frac = np.zeros((m, HAND_SIZE), dtype=np.float32)
        in_gap_frac = np.zeros((m, HAND_SIZE), dtype=np.float32)

    is_sixth = (~below_all) & (t_len == ROW_CAP)
    take_now = np.where(
        below_all, np.broadcast_to(min_bull[:, None], is_sixth.shape),
        np.where(is_sixth, t_bull, 0.0),
    ).astype(np.float32)
    gap = np.where(
        below_all, 0.0, hands.astype(np.float32) - t_end.astype(np.float32)
    ).astype(np.float32)

    return {
        "mask": mask, "cards": cards, "tgt": tgt, "below_all": below_all,
        "is_sixth": is_sixth, "t_len": t_len, "t_bull": t_bull, "gap": gap,
        "take_now": take_now, "min_bull": min_bull,
        "lower_frac": lower_frac,
        "in_gap_frac": in_gap_frac,
    }


def card_features(env, games, players):
    i = placement_info(env, games, players)
    m = i["mask"].shape[0]
    hands = i["cards"].astype(np.float32)
    # below_all のとき tgt は argmax の既定値 0 を指す。行き先に依存する特徴を0にして、
    # 行0の状態が「行き先の状態」として紛れ込むのを防ぐ
    has_target = (~i["below_all"]).astype(np.float32)
    f = np.zeros((m, HAND_SIZE, C_DIM), dtype=np.float32)
    f[:, :, 0] = hands / np.float32(N_CARDS)
    f[:, :, 1] = BULL[i["cards"]].astype(np.float32) / 7.0
    f[:, :, 2] = i["t_len"] / np.float32(ROW_CAP) * has_target
    f[:, :, 3] = i["t_bull"] / _BULL_SCALE * has_target
    f[:, :, 4] = i["gap"] / np.float32(N_CARDS)
    f[:, :, 5] = i["is_sixth"]
    f[:, :, 6] = i["below_all"]
    f[:, :, 7] = (ROW_CAP - i["t_len"]) / np.float32(ROW_CAP) * has_target
    f[:, :, 8] = i["take_now"] / _BULL_SCALE
    f[:, :, 9] = i["lower_frac"]
    f[:, :, 10] = i["in_gap_frac"]
    f[:, :, 11] = np.arange(HAND_SIZE, dtype=np.float32) / HAND_SIZE
    f *= i["mask"][:, :, None]
    return f, i["mask"]


def card_obs(env):
    B, P = env.B, env.P
    games = np.repeat(np.arange(B, dtype=np.intp), P)
    players = np.tile(np.arange(P, dtype=np.intp), B)
    g = global_features(env, games, players).reshape(B, P, G_DIM)
    c, mask = card_features(env, games, players)
    return g, c.reshape(B, P, HAND_SIZE, C_DIM), mask.reshape(B, P, HAND_SIZE)


def row_obs(env):
    games = env.pending_games
    players = env.pending_players
    cards = env.pending_cards
    g = global_features(env, games, players, played_cards=cards)

    m = games.size
    ends = env.row_ends()[games].astype(np.float32)
    lens = env.row_len[games].astype(np.intp)
    bulls = env.row_points_all(games).astype(np.float32)
    is_min = (bulls == bulls.min(axis=1, keepdims=True)).astype(np.float32)

    r = np.zeros((m, N_ROWS, R_DIM), dtype=np.float32)
    r[:, :, 0] = lens / np.float32(ROW_CAP)
    r[:, :, 1] = bulls / _BULL_SCALE
    r[:, :, 2] = ends / np.float32(N_CARDS)
    r[:, :, 3:8] = _ONEHOT5[lens - 1]
    r[:, :, 8] = is_min
    return g, r
