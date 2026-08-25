"""方策・価値ネットワーク。手札の各カードを個別にスコアリングする。"""

import torch
import torch.nn as nn

from ..features import C_DIM, G_DIM, R_DIM

NEG_INF = float("-inf")


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class NimmtNet(nn.Module):
    def __init__(self, g_dim: int = G_DIM, c_dim: int = C_DIM, r_dim: int = R_DIM,
                 hidden: int = 256, depth: int = 2):
        super().__init__()
        self.g_dim, self.c_dim, self.r_dim = g_dim, c_dim, r_dim
        self.hidden, self.depth = hidden, depth

        enc: list[nn.Module] = [nn.Linear(g_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)]
        for _ in range(depth - 1):
            enc += [nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)]
        self.enc = nn.Sequential(*enc)

        self.card_head = _mlp([hidden + c_dim, hidden // 2, 1])
        self.row_head = _mlp([hidden + r_dim, hidden // 2, 1])
        self.value_head = _mlp([hidden, hidden // 2, 1])

    def encode(self, g: torch.Tensor) -> torch.Tensor:
        return self.enc(g)

    def card_logits(self, h, cards, mask):
        n, k, _ = cards.shape
        x = torch.cat([h[:, None, :].expand(n, k, self.hidden), cards], dim=2)
        lg = self.card_head(x).squeeze(-1)
        return lg.masked_fill(~mask, NEG_INF)

    def row_logits(self, h, rows):
        n, k, _ = rows.shape
        x = torch.cat([h[:, None, :].expand(n, k, self.hidden), rows], dim=2)
        return self.row_head(x).squeeze(-1)

    def value(self, h) -> torch.Tensor:
        return self.value_head(h).squeeze(-1)

    def config(self) -> dict:
        return {"g_dim": self.g_dim, "c_dim": self.c_dim, "r_dim": self.r_dim,
                "hidden": self.hidden, "depth": self.depth}


def save(net: NimmtNet, path) -> None:
    torch.save({"config": net.config(), "state_dict": net.state_dict()}, path)


def load(path, device: str = "cpu") -> NimmtNet:
    blob = torch.load(path, map_location=device, weights_only=False)
    net = NimmtNet(**blob["config"])
    net.load_state_dict(blob["state_dict"])
    net.to(device).eval()
    return net
