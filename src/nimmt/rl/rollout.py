"""自己対戦とリーグ対戦の経験収集。"""

import numpy as np
import torch

from ..features import card_features, global_features, row_obs
from ..vec import PHASE_ROW

REWARD_SCALE = 10.0

# 学習用の経験収集では temperature を 1.0 から動かさないこと。
# `_sample` は softmax(logits / T) の log 確率を保存するが、`ppo._pass` は
# Categorical(logits=logits)（= T=1）で再計算するため、T != 1 だと epoch 0 の
# 重要度比が 1 から始まらず、clip 帯が最初からずれる。
# 実測: 方策が鋭くなったネット・T=0.5 で 11.4% のサンプルが最初から clip 帯の外に出る
# （＝そのサンプルの勾配が消える）。探索は entropy 係数で行うので temperature は不要。
# 推論・評価用の NeuralBot 側の temperature はこの制約と無関係。


def _t(a, device, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=device)


def _sample(logits: np.ndarray, rng, temperature: float):
    """Gumbel-max でサンプリングし、その行動の log 確率を返す。

    「累積和と一様乱数を比べる」方式は使わない。float32 の累積和は最終値が 1.0 を
    わずかに下回ることがあり、u がそこを超えると全比較が True になって
    最後のスロット＝マスクされた無効な手に落ちる（実測で学習1ランあたり期待5.5回）。
    Gumbel-max は -inf を保つのでこの経路が原理的に存在しない。
    log 確率は logsumexp で安定に計算する。
    """
    z = logits / max(temperature, 1e-6)
    g = rng.gumbel(size=z.shape).astype(np.float32)
    a = (z + g).argmax(axis=1).astype(np.int64)
    zmax = z.max(axis=1, keepdims=True)
    logsumexp = zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))
    logp = z[np.arange(z.shape[0]), a] - logsumexp
    return a, logp.astype(np.float32)


@torch.no_grad()
def _act_cards(net, env, games, players, device, rng, temperature):
    g = global_features(env, games, players)
    c, mask = card_features(env, games, players)
    h = net.encode(_t(g, device))
    lg = net.card_logits(h, _t(c, device), _t(mask, device, torch.bool)).cpu().numpy()
    v = net.value(h).cpu().numpy()
    a, logp = _sample(lg, rng, temperature)
    return a, logp, v, g, c, mask


@torch.no_grad()
def _act_rows(net, env, sel, device, rng, temperature):
    g_all, r_all = row_obs(env)
    g, r = g_all[sel], r_all[sel]
    h = net.encode(_t(g, device))
    lg = net.row_logits(h, _t(r, device)).cpu().numpy()
    v = net.value(h).cpu().numpy()
    a, logp = _sample(lg, rng, temperature)
    return a, logp, v, g, r


def collect_selfplay(net, env, buf, device="cpu", temperature=1.0, rng=None, reward_mode="absolute"):
    rng = rng if rng is not None else np.random.default_rng(0)
    B, P = env.B, env.P
    games = np.repeat(np.arange(B, dtype=np.intp), P)
    players = np.tile(np.arange(P, dtype=np.intp), B)
    keys = games * P + players

    while not env.done():
        a, logp, v, g, c, mask = _act_cards(net, env, games, players, device, rng, temperature)
        buf.add_card(keys, g, c, mask, a, logp, v)
        env.step_cards(a.reshape(B, P))
        while env.phase == PHASE_ROW:
            sel = np.arange(env.pending_games.size, dtype=np.intp)
            ra, rlogp, rv, rg, rx = _act_rows(net, env, sel, device, rng, temperature)
            rkeys = env.pending_games * P + env.pending_players
            buf.add_row(rkeys, rg, rx, ra, rlogp, rv)
            env.step_rows(ra)
        pts = env.last_turn_points().astype(np.float32)             # (B,P)
        if reward_mode == "relative":
            total = pts.sum(axis=1, keepdims=True)
            others_mean = (total - pts) / max(P - 1, 1)
            r = -(pts - others_mean)
        else:
            r = -pts
        buf.add_reward(keys, (r.ravel() / REWARD_SCALE))
    return env.taken.copy()


def collect_league(net, opp_net, env, buf, device="cpu", learner_seat=0, temperature=1.0,
                   rng=None, key_offset=0, reward_mode="absolute"):
    """learner_seat のみ net で打ち、経験を記録する。他席は opp_net（記録しない）。

    key_offset は、同じバッファに自己対戦分とリーグ分を入れるときのキー衝突を避けるための
    ゲーム index のずらし幅。自己対戦を n_self ゲーム入れたなら key_offset=n_self を渡す。"""
    rng = rng if rng is not None else np.random.default_rng(0)
    B, P = env.B, env.P
    gidx = np.arange(B, dtype=np.intp)
    my_players = np.full(B, learner_seat, dtype=np.intp)
    keys = (gidx + key_offset) * P + my_players

    while not env.done():
        slots = np.zeros((B, P), dtype=np.intp)
        a, logp, v, g, c, mask = _act_cards(net, env, gidx, my_players, device, rng, temperature)
        buf.add_card(keys, g, c, mask, a, logp, v)
        slots[:, learner_seat] = a
        for s in range(P):
            if s == learner_seat:
                continue
            oa, _, _, _, _, _ = _act_cards(
                opp_net, env, gidx, np.full(B, s, np.intp), device, rng, temperature
            )
            slots[:, s] = oa
        env.step_cards(slots)

        while env.phase == PHASE_ROW:
            choices = np.zeros(env.pending_games.size, dtype=np.intp)
            mine = np.flatnonzero(env.pending_players == learner_seat)
            if mine.size:
                ra, rlogp, rv, rg, rx = _act_rows(net, env, mine, device, rng, temperature)
                buf.add_row((env.pending_games[mine] + key_offset) * P + learner_seat,
                            rg, rx, ra, rlogp, rv)
                choices[mine] = ra
            other = np.flatnonzero(env.pending_players != learner_seat)
            if other.size:
                oa, _, _, _, _ = _act_rows(opp_net, env, other, device, rng, temperature)
                choices[other] = oa
            env.step_rows(choices)

        pts = env.last_turn_points()
        mine = pts[gidx, learner_seat].astype(np.float32)
        if reward_mode == "relative":
            others = (pts.sum(axis=1).astype(np.float32) - mine) / max(P - 1, 1)
            r = -(mine - others)
        else:
            r = -mine
        buf.add_reward(keys, r / REWARD_SCALE)
    return env.taken.copy()
