"""Web(GitHub Pages)用に学習済みモデルとゴールデンテストベクトルを書き出す。

- pages/weights_{name}.json : {config, tensors: {state_dict名: {shape, b64(float32 LE)}}}
- pages/golden.json          : ランダム局面の (盤面, 特徴量, logits)。JS移植の数値照合用

Usage: uv run python scripts/export_web.py
"""

import base64
import json
from pathlib import Path

import numpy as np
import torch

from nimmt.features import card_features, global_features, row_obs
from nimmt.rl.model import load
from nimmt.vec import PHASE_ROW, VecNimmt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages"
OUT.mkdir(exist_ok=True)


def export_net(name: str, ckpt: str):
    net = load(ROOT / ckpt)
    tensors = {}
    for k, v in net.state_dict().items():
        a = v.numpy().astype("<f4")
        tensors[k] = {"shape": list(a.shape),
                      "b64": base64.b64encode(a.tobytes()).decode()}
    blob = {"config": net.config(), "tensors": tensors}
    p = OUT / f"weights_{name}.json"
    p.write_text(json.dumps(blob))
    print(f"{p.name}: {p.stat().st_size // 1024} KB")
    return net


def snap_env(env):
    """JS側でそのまま局面を再構築できる形にする。"""
    return {
        "rows": [[int(c) for c in env.rows[0, r, : env.row_len[0, r]]] for r in range(4)],
        "hands": [[int(c) for c in env.hands[0, p] if c > 0] for p in range(env.P)],
        "hand_slots": [[int(c) for c in env.hands[0, p]] for p in range(env.P)],
        "taken": env.taken[0].astype(int).tolist(),
        "seen": [int(c) for c in np.flatnonzero(env.seen[0])],
        "turn": int(env.turn),
        "players": env.P,
    }


def golden(net4, net2, n_cases=30, seed=7):
    rng = np.random.default_rng(seed)
    cases = []
    for players, net in ((4, net4), (2, net2)):
        env = VecNimmt(1, players, seed=int(rng.integers(1 << 30)))
        while len([c for c in cases if c["players"] == players]) < n_cases:
            if env.done():
                env = VecNimmt(1, players, seed=int(rng.integers(1 << 30)))
            seat = int(rng.integers(players))
            g0 = np.array([0], dtype=np.intp)
            ps = np.array([seat], dtype=np.intp)
            g = global_features(env, g0, ps)
            c, mask = card_features(env, g0, ps)
            with torch.no_grad():
                h = net.encode(torch.as_tensor(g))
                lg = net.card_logits(h, torch.as_tensor(c),
                                     torch.as_tensor(mask)).numpy()
            finite = lg[0][np.isfinite(lg[0])]
            cases.append({
                "kind": "card", "players": players, "seat": seat,
                "state": snap_env(env),
                "g": g[0].tolist(),
                "logits": [x if np.isfinite(x) else None for x in lg[0].tolist()],
                "argmax": int(np.nanargmax(np.where(np.isfinite(lg[0]), lg[0], -1e30))),
            })
            # 進める（全席ランダム）。PHASE_ROW はランダム行で解決しつつ row ケースも収集
            slots = np.zeros((1, players), dtype=np.intp)
            for p in range(players):
                avail = np.flatnonzero(env.hands[0, p] > 0)
                slots[0, p] = int(rng.choice(avail))
            env.step_cards(slots)
            while env.phase == PHASE_ROW:
                gg, rr = row_obs(env)
                with torch.no_grad():
                    h = net.encode(torch.as_tensor(gg))
                    rlg = net.row_logits(h, torch.as_tensor(rr)).numpy()
                cases.append({
                    "kind": "row", "players": players,
                    "seat": int(env.pending_players[0]),
                    "pending_card": int(env.pending_cards[0]),
                    "state": snap_env(env),
                    "g": gg[0].tolist(), "r": rr[0].tolist(),
                    "logits": rlg[0].tolist(),
                    "argmax": int(rlg[0].argmax()),
                })
                env.step_rows(np.array([int(rng.integers(4))], dtype=np.intp))
    (OUT / "golden.json").write_text(json.dumps(cases))
    print(f"golden.json: {len(cases)} cases")


net4 = export_net("exp4", "runs/exp4/latest.pt")
net2 = export_net("exp2", "runs/exp6/latest.pt")   # 2人戦モデルの実体は exp6（web側の名前は据え置き）
golden(net4, net2)
