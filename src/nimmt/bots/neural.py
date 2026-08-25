"""学習済みネットを Bot インターフェースに載せるラッパ。評価と対人対戦で使う。"""

import numpy as np
import torch

from ..features import card_features, global_features, row_obs
from .base import Bot


class NeuralBot(Bot):
    name = "neural"

    def __init__(self, net, device: str = "cpu", temperature: float = 0.0, seed: int = 0):
        self.net = net.to(device).eval()
        self.device = device
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)

    def _sample(self, logits: np.ndarray) -> np.ndarray:
        """Gumbel-max によるカテゴリカルサンプリング。

        「累積和と一様乱数を比べる」方式は使わない。float32 の累積和は最終値が 1.0 を
        わずかに下回ることがあり（実測: 有効10スロットで平均 2.3e-8、最大 3.0e-7 の不足）、
        u がそこを超えると全比較が True になって最後のスロット＝**マスクされた無効な手**に
        落ちる。学習1ラン（約2.5億決定）で期待5.5回発火し、そのスロットが既出カードなら
        `step_cards` の assert が飛んで数時間の学習が死ぬ。
        Gumbel-max は -inf をそのまま -inf に保つので、この経路が原理的に存在しない。
        """
        if self.temperature <= 0.0:
            return logits.argmax(axis=1)
        g = self.rng.gumbel(size=logits.shape).astype(np.float32)
        return (logits / np.float32(self.temperature) + g).argmax(axis=1)

    @torch.no_grad()
    def pick_cards(self, env, games, players):
        g = global_features(env, games, players)
        c, mask = card_features(env, games, players)
        t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=self.device)
        h = self.net.encode(t(g))
        lg = self.net.card_logits(h, t(c), t(mask, torch.bool)).cpu().numpy()
        return self._sample(lg)

    @torch.no_grad()
    def pick_rows(self, env, sel):
        sel = np.asarray(sel, dtype=np.intp)
        g_all, r_all = row_obs(env)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        h = self.net.encode(t(g_all[sel]))
        lg = self.net.row_logits(h, t(r_all[sel])).cpu().numpy()
        return self._sample(lg)
