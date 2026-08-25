"""PPO 更新。カード相と行相を別パスで更新し、価値損失は両方に含める。"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    # 学習率は optimizer 側で設定する。ここには置かない（二重管理になって片方が死ぬ）
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 8192
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    max_grad_norm: float = 0.5


def _pass(net, opt, cfg, device, g, x, mask, a, old_logp, adv, ret, is_card, stats, rng):
    n = g.shape[0]
    if n == 0:
        return
    order = rng.permutation(n)
    for start in range(0, n, cfg.minibatch):
        mb = order[start : start + cfg.minibatch]
        tg = torch.as_tensor(g[mb], dtype=torch.float32, device=device)
        tx = torch.as_tensor(x[mb], dtype=torch.float32, device=device)
        ta = torch.as_tensor(a[mb], dtype=torch.int64, device=device)
        tl = torch.as_tensor(old_logp[mb], dtype=torch.float32, device=device)
        tadv = torch.as_tensor(adv[mb], dtype=torch.float32, device=device)
        tret = torch.as_tensor(ret[mb], dtype=torch.float32, device=device)

        h = net.encode(tg)
        if is_card:
            tm = torch.as_tensor(mask[mb], dtype=torch.bool, device=device)
            logits = net.card_logits(h, tx, tm)
        else:
            logits = net.row_logits(h, tx)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(ta)
        entropy = dist.entropy().mean()

        ratio = torch.exp(logp - tl)
        pl = -torch.min(ratio * tadv,
                        torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * tadv).mean()
        vl = F.mse_loss(net.value(h), tret)
        loss = pl + cfg.vf_coef * vl - cfg.ent_coef * entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        opt.step()

        with torch.no_grad():
            stats["policy_loss"].append(float(pl))
            stats["value_loss"].append(float(vl))
            stats["entropy"].append(float(entropy))
            stats["clip_frac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
            stats["approx_kl"].append(float((tl - logp).mean()))


def ppo_update(net, opt, batch, cfg: PPOConfig, device: str = "cpu", rng=None) -> dict:
    rng = rng if rng is not None else np.random.default_rng(0)
    net.train()
    stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl")}
    for _ in range(cfg.epochs):
        _pass(net, opt, cfg, device, batch.card_g, batch.card_x, batch.card_mask,
              batch.card_a, batch.card_logp, batch.card_adv, batch.card_ret, True, stats, rng)
        _pass(net, opt, cfg, device, batch.row_g, batch.row_x, None,
              batch.row_a, batch.row_logp, batch.row_adv, batch.row_ret, False, stats, rng)
    net.eval()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
