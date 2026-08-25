"""exp1（絶対報酬）と exp3（相対報酬）の最終方策を直接対戦させる。
Usage: uv run python scripts/ablation.py --run-a runs/exp1 --run-b runs/exp3 --deals 20000

[A, B, A, B] の交互着席で対戦し、各ディールの A側2席の平均失点 − B側2席の平均失点を
ペア差としてブートストラップCIを出す。差が負なら A の方が失点が少ない（＝強い）。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from nimmt.arena import _boot_ci, run_match
from nimmt.bots.neural import NeuralBot
from nimmt.rl.model import load
from nimmt.vec import new_decks

EVAL_SEED = 20260901  # report.py の評価配牌ともモデル選択の配牌とも別のシード


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-a", required=True)
    p.add_argument("--run-b", required=True)
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--deals", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="docs/ablation.json")
    args = p.parse_args(argv)

    net_a = load(Path(args.run_a) / args.checkpoint, device=args.device)
    net_b = load(Path(args.run_b) / args.checkpoint, device=args.device)
    decks = new_decks(args.deals, np.random.default_rng(EVAL_SEED))
    bots = [NeuralBot(net_a, device=args.device), NeuralBot(net_b, device=args.device),
            NeuralBot(net_a, device=args.device), NeuralBot(net_b, device=args.device)]
    pd = run_match(bots, decks).astype(np.float64)
    a = (pd[:, 0] + pd[:, 2]) / 2.0
    b = (pd[:, 1] + pd[:, 3]) / 2.0
    diff, lo, hi = _boot_ci(a - b, n_boot=10000, seed=0)
    result = {
        "run_a": args.run_a, "run_b": args.run_b, "n_deals": int(pd.shape[0]),
        "a_mean_loss": float(a.mean()), "b_mean_loss": float(b.mean()),
        "diff_a_minus_b": diff, "ci_low": lo, "ci_high": hi,
        "a_win_rate": float(((a < b).sum() + 0.5 * (a == b).sum()) / pd.shape[0]),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
