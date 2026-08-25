"""速度ベンチ。env のスループット、CPU vs MPS のネット速度、1イテレーションの実測。

**1イテレーションの計測は必ず複数回繰り返す。**単発の計測でマシンの背景負荷を拾うと
device の選択が実際とは逆になる（実測: 同一マシンで単発では CPU が 1.52倍速く見えたが、
6回繰り返すと 6/6 で MPS が同等以上だった）。中央値と全実測値の両方を出し、
判断材料として load average も記録する。

Usage: uv run python scripts/bench.py > bench.json"""

import json
import os
import time

import numpy as np
import torch

from nimmt.bots.simple import HeuristicBot
from nimmt.features import C_DIM, G_DIM, R_DIM, card_obs
from nimmt.rl.buffer import TrajectoryBuffer
from nimmt.rl.model import NimmtNet
from nimmt.rl.ppo import PPOConfig, ppo_update
from nimmt.rl.rollout import collect_selfplay
from nimmt.vec import PHASE_ROW, VecNimmt


def bench_env(batch: int, players: int = 4) -> dict:
    """env + HeuristicBot のスループット。env 単体ではなく bot の思考を含む。"""
    bot = HeuristicBot()
    env = VecNimmt(batch, players, seed=0)
    games = np.repeat(np.arange(batch), players)
    plyrs = np.tile(np.arange(players), batch)
    n_row = 0
    t0 = time.perf_counter()
    while not env.done():
        env.step_cards(bot.pick_cards(env, games, plyrs).reshape(batch, players))
        while env.phase == PHASE_ROW:
            n_row += int(env.pending_games.size)
            env.step_rows(bot.pick_rows(env, np.arange(env.pending_games.size)))
    dt = time.perf_counter() - t0
    n_card = batch * players * 10
    return {"batch": batch, "sec_per_deal_batch": dt,
            "deals_per_sec": batch / dt,
            "card_decisions": n_card, "row_decisions": n_row,
            "decisions_per_sec": (n_card + n_row) / dt}


def bench_net(device: str, n: int, hidden: int = 256) -> dict | None:
    if device == "mps" and not torch.backends.mps.is_available():
        return None
    net = NimmtNet(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    g = torch.randn(n, G_DIM, device=device)
    c = torch.randn(n, 10, C_DIM, device=device)
    m = torch.ones(n, 10, dtype=torch.bool, device=device)
    tgt = torch.randn(n, device=device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(3):
        h = net.encode(g)
        net.card_logits(h, c, m).sum().backward()
        opt.zero_grad(set_to_none=True)
    sync()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            h = net.encode(g)
            net.card_logits(h, c, m)
            net.value(h)
    sync()
    fwd = (time.perf_counter() - t0) / 20

    t0 = time.perf_counter()
    for _ in range(20):
        h = net.encode(g)
        loss = net.card_logits(h, c, m).sum() + ((net.value(h) - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    bwd = (time.perf_counter() - t0) / 20
    return {"device": device, "n": n, "hidden": hidden, "fwd_sec": fwd, "bwd_sec": bwd}


def bench_iteration(device: str, games: int = 2048, players: int = 4, reps: int = 5) -> dict:
    """1イテレーション（経験収集 + PPO更新）を reps 回計測して中央値を返す。

    単発計測は背景負荷を拾って device の選択を逆にしうるので、必ず繰り返す。
    最初の1回はウォームアップとして捨てる（初回のカーネルコンパイルとメモリ確保を除くため。
    ウォームアップは row_logits も通るので、そのコンパイルコストも本計測から外れる）。
    """
    net = NimmtNet(hidden=256, depth=2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)

    def one() -> tuple[float, float]:
        env = VecNimmt(games, players, seed=0)
        buf = TrajectoryBuffer(n_keys=games * players)
        t0 = time.perf_counter()
        collect_selfplay(net, env, buf, device=device, rng=rng)
        t1 = time.perf_counter()
        ppo_update(net, opt, buf.finish(), PPOConfig(), device=device, rng=rng)
        if device == "mps":
            torch.mps.synchronize()
        return t1 - t0, time.perf_counter() - t1

    one()  # ウォームアップ（捨てる）
    load_before = os.getloadavg()
    runs = [one() for _ in range(reps)]
    load_after = os.getloadavg()

    def med(xs):
        return sorted(xs)[len(xs) // 2]

    totals = [c + u for c, u in runs]
    return {"device": device, "games": games, "reps": reps,
            "collect_sec_median": med([c for c, _ in runs]),
            "update_sec_median": med([u for _, u in runs]),
            "total_sec_median": med(totals),
            "total_sec_all": totals,
            "games_per_sec_median": games / med(totals),
            "loadavg_before": load_before, "loadavg_after": load_after}


def main() -> int:
    out = {
        "env": [bench_env(b) for b in (512, 2048, 8192)],
        "net": [r for d in ("cpu", "mps") for n in (2048, 8192, 32768)
                if (r := bench_net(d, n)) is not None],
        "iteration": [r for d in ("cpu", "mps")
                      if (torch.backends.mps.is_available() or d == "cpu")
                      for r in [bench_iteration(d)]],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
