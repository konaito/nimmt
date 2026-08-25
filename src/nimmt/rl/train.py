"""PPO 自己対戦＋リーグの学習エントリポイント。

Usage:
  uv run python -m nimmt.rl.train --run-dir runs/exp1 --iterations 3000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..arena import run_match, summarize
from ..bots.neural import NeuralBot
from ..bots.simple import GreedyBot, HeuristicBot, RandomBot
from ..vec import VecNimmt, new_decks
from .buffer import TrajectoryBuffer
from .league import CheckpointPool
from .model import NimmtNet, load as model_load, save
from .ppo import PPOConfig, ppo_update
from .rollout import collect_league, collect_selfplay

# 学習中のモニタリング（＝実質モデル選択）に使う配牌シード。
# 最終レポート（scripts/report.py の EVAL_SEED）とは必ず別にすること。
# 同じ配牌で見出し数字を出すと、モデル選択に使った盤面で自己採点することになる。
MONITOR_SEED = 20260824


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--games", type=int, default=2048)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--ent-start", type=float, default=0.02)
    p.add_argument("--ent-end", type=float, default=0.002)
    p.add_argument("--ent-decay-frac", type=float, default=0.6)
    p.add_argument("--league-frac", type=float, default=0.5)
    p.add_argument("--pool-every", type=int, default=50)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--reward-mode", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--init-from", type=str, default=None,
                   help="既存チェックポイントから重みを引き継ぐ。hidden/depth は"
                        "チェックポイント側の値が使われ、--hidden/--depth は無視される")
    # 学習用の収集で temperature を 1.0 から動かすフラグは置かない。
    # `_sample` は softmax(logits/T) の log 確率を保存するが `ppo._pass` は T=1 で再計算するため、
    # T != 1 だと epoch 0 の重要度比が 1 から始まらず clip 帯が最初からずれる
    # （実測: 鋭い方策・T=0.5 で 11.4% のサンプルが最初から clip 帯の外＝勾配が消える）。
    # 探索は entropy 係数の減衰で行う。
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-deals", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=100)
    return p


def evaluate(net, players: int, n_deals: int, device: str) -> dict:
    """固定シードの重複配牌で heuristic / greedy / random と対戦させる。"""
    decks = new_decks(n_deals, np.random.default_rng(MONITOR_SEED))
    out = {}
    for opp in (HeuristicBot(), GreedyBot(), RandomBot(seed=0)):
        bots = [NeuralBot(net, device=device)] + [opp] * (players - 1)
        r = summarize(run_match(bots, decks), ["neural"] + [opp.name] * (players - 1))
        out[f"eval_{opp.name}"] = float(r.mean_loss[0])
        out[f"eval_{opp.name}_opp"] = float(r.mean_loss[1:].mean())
        out[f"eval_{opp.name}_winrate"] = float(r.win_rate[0])
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    if args.init_from:
        # load() は eval モードで返すが、NimmtNet は LayerNorm のみで
        # train/eval で挙動は変わらない。新規初期化と揃えるため train に戻す。
        net = model_load(args.init_from, device=device).train()
        args.hidden, args.depth = net.hidden, net.depth
    else:
        net = NimmtNet(hidden=args.hidden, depth=args.depth).to(device)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    pool = CheckpointPool(max_size=args.pool_size)
    pool.add(net)

    log = (run / "log.jsonl").open("a")
    n_league = int(args.games * args.league_frac)
    n_self = args.games - n_league

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        frac = min(1.0, it / max(1, args.iterations * args.ent_decay_frac))
        ent = args.ent_start + (args.ent_end - args.ent_start) * frac
        cfg = PPOConfig(clip=args.clip, epochs=args.epochs,
                        minibatch=args.minibatch, ent_coef=ent)

        buf = TrajectoryBuffer(n_keys=args.games * args.players)
        losses = []

        if n_self > 0:
            env = VecNimmt(n_self, args.players, seed=int(rng.integers(1 << 30)))
            losses.append(collect_selfplay(net, env, buf, device=device,
                                           temperature=1.0, rng=rng,
                                           reward_mode=args.reward_mode).mean())
        if n_league > 0 and len(pool) > 0:
            opp = pool.sample(rng, net).to(device)
            env = VecNimmt(n_league, args.players, seed=int(rng.integers(1 << 30)))
            seat = int(rng.integers(args.players))
            taken = collect_league(net, opp, env, buf, device=device, learner_seat=seat,
                                   temperature=1.0, rng=rng,
                                   reward_mode=args.reward_mode, key_offset=n_self)
            losses.append(taken[:, seat].mean())

        # rng を渡さないと ppo_update が毎回 default_rng(0) を作り直し、
        # ミニバッチのシャッフル順が全イテレーションで厳密に同一になる
        metrics = ppo_update(net, opt, buf.finish(), cfg, device=device, rng=rng)
        dt = time.perf_counter() - t0
        rec = {"iter": it, "mean_loss": float(np.mean(losses)), "ent_coef": ent,
               "sec": dt, "games_per_sec": args.games / dt, **metrics}

        if args.eval_every and it % args.eval_every == 0:
            rec.update(evaluate(net, args.players, args.eval_deals, device))
        if args.pool_every and it % args.pool_every == 0:
            pool.add(net)
        if args.ckpt_every and it % args.ckpt_every == 0:
            save(net, run / f"ckpt_{it:06d}.pt")
        save(net, run / "latest.pt")

        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
