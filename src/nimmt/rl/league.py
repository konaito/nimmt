"""対戦相手プール。純粋な自己対戦だけだと方策が循環して脆くなるため、
過去のチェックポイントを混ぜる。"""

import copy

import numpy as np


class CheckpointPool:
    def __init__(self, max_size: int = 20):
        self.max_size = int(max_size)
        self._items: list[dict] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, net) -> None:
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        self._items.append(sd)
        if len(self._items) > self.max_size:
            self._items.pop(0)

    def sample(self, rng: np.random.Generator, template_net):
        assert self._items, "プールが空"
        sd = self._items[int(rng.integers(len(self._items)))]
        opp = copy.deepcopy(template_net)
        opp.load_state_dict(sd)
        opp.eval()
        for p in opp.parameters():
            p.requires_grad_(False)
        return opp
