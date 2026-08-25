"""学習結果の評価レポートを作る。
Usage: uv run python scripts/report.py --run-dir runs/exp1 --deals 20000"""

import argparse
import json
from pathlib import Path

import numpy as np

from nimmt.arena import paired_ci_field, run_match, summarize
from nimmt.bots.neural import NeuralBot
from nimmt.bots.rollout import MCRolloutBot
from nimmt.bots.simple import GreedyBot, HeuristicBot, RandomBot
from nimmt.elo import bradley_terry
from nimmt.rl.model import load
from nimmt.vec import new_decks

# 学習中のモニタリング（= 実質モデル選択）に使った配牌とは別のシードを使う。
# 同じ配牌で最終レポートの見出し数字を出すと、選んだ盤面で自己採点することになる。
EVAL_SEED = 20260831


def vs_baselines(net, players: int, deals: int, device: str, skip_rollout: bool,
                 rollout_deals: int = 4000, rollout_samples: int = 24) -> list[dict]:
    """固定シードの重複配牌でベースライン群と対戦させる。

    mc-rollout だけ `n_samples` を上げ、代わりにディール数を減らす。既定の n_samples=8 は
    バグではなく分散で頭打ちになっており（実測: n_samples=1/2/8/24 で
    mc − heuristic = +1.048 / +0.679 / −1.276 / −2.697 と単調改善）、
    弱い mc-rollout を相手にすると完了条件のゲートが不当に緩くなる。
    1手あたり n_samples×top_k ゲームのプレイアウトが走るので、ディール数で釣り合わせる。
    """
    decks = new_decks(deals, np.random.default_rng(EVAL_SEED))
    matchups = [(RandomBot(seed=0), decks), (GreedyBot(), decks), (HeuristicBot(), decks)]
    if not skip_rollout:
        matchups.append(
            (MCRolloutBot(n_samples=rollout_samples, top_k=4, seed=0),
             decks[: min(rollout_deals, deals)])
        )
    rows = []
    for opp, opp_decks in matchups:
        bots = [NeuralBot(net, device=device)] + [opp] * (players - 1)
        per_deal = run_match(bots, opp_decks)
        r = summarize(per_deal, ["neural"] + [opp.name] * (players - 1))
        opp_mean = float(per_deal[:, 1:].mean())
        diff, lo, hi = paired_ci_field(per_deal, 0, n_boot=10000, seed=0)
        rows.append({
            "opponent": opp.name, "neural_mean_loss": float(r.mean_loss[0]),
            "opp_mean_loss": opp_mean, "diff": diff, "ci_low": lo, "ci_high": hi,
            "neural_win_rate": float(r.win_rate[0]), "n_deals": int(per_deal.shape[0]),
        })
    return rows


def checkpoint_elo(run: Path, deals: int, device: str, max_ckpts: int = 10) -> dict:
    """チェックポイント同士の総当たり。ペア数が O(n^2) なので等間隔に max_ckpts 個へ間引く。"""
    ckpts = sorted(run.glob("ckpt_*.pt"))
    if len(ckpts) < 2:
        return {"names": [], "ratings": []}
    if len(ckpts) > max_ckpts:
        pick = np.linspace(0, len(ckpts) - 1, max_ckpts).round().astype(int)
        ckpts = [ckpts[i] for i in dict.fromkeys(pick.tolist())]
    names = [c.stem for c in ckpts]
    nets = [load(c, device=device) for c in ckpts]
    decks = new_decks(deals, np.random.default_rng(EVAL_SEED + 1))
    n = len(nets)
    wins = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            bots = [NeuralBot(nets[i], device=device), NeuralBot(nets[j], device=device),
                    NeuralBot(nets[i], device=device), NeuralBot(nets[j], device=device)]
            pd = run_match(bots, decks)
            a = (pd[:, 0].astype(float) + pd[:, 2]) / 2.0
            b = (pd[:, 1].astype(float) + pd[:, 3]) / 2.0
            wins[i, j] = (a < b).sum() + 0.5 * (a == b).sum()
            wins[j, i] = (b < a).sum() + 0.5 * (a == b).sum()
    return {"names": names, "ratings": bradley_terry(wins).tolist()}


def to_markdown(data: dict) -> str:
    lines = ["# 6 Nimmt! 学習結果レポート", ""]
    for key, title in (("baselines_4p", "4人戦"), ("baselines_2p", "2人戦")):
        lines += [f"## {title}（対ベースライン）", "",
                  "| 相手 | AI平均失点 | 相手平均失点 | 差 | 95%CI | AI勝率 | ディール数 |",
                  "|---|---|---|---|---|---|---|"]
        for r in data[key]:
            lines.append(
                f"| {r['opponent']} | {r['neural_mean_loss']:.3f} | {r['opp_mean_loss']:.3f} | "
                f"{r['diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                f"{r['neural_win_rate']:.3f} | {r['n_deals']} |"
            )
        lines.append("")
    lines += ["## チェックポイントの Elo", "", "| チェックポイント | Elo |", "|---|---|"]
    for name, rating in zip(data["elo"]["names"], data["elo"]["ratings"]):
        lines.append(f"| {name} | {rating:+.1f} |")
    lines += ["", "差が負なら AI の方が失点が少ない（＝強い）。95%CI が 0 を跨がなければ有意。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--deals", type=int, default=20000)
    p.add_argument("--rollout-deals", type=int, default=4000,
                   help="mc-rollout との対戦だけこのディール数に減らす（1手あたり多数のプレイアウトが走るため）")
    p.add_argument("--rollout-samples", type=int, default=24,
                   help="mc-rollout の相手手札サンプル数。既定の8では分散で頭打ちになりゲートが緩くなる")
    p.add_argument("--elo-deals", type=int, default=2000)
    p.add_argument("--max-elo-ckpts", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-rollout", action="store_true")
    args = p.parse_args(argv)

    run = Path(args.run_dir)
    net = load(run / args.checkpoint, device=args.device)
    data = {
        "checkpoint": args.checkpoint,
        "baselines_4p": vs_baselines(net, 4, args.deals, args.device, args.skip_rollout,
                                     args.rollout_deals, args.rollout_samples),
        "baselines_2p": vs_baselines(net, 2, args.deals, args.device, args.skip_rollout,
                                     args.rollout_deals, args.rollout_samples),
        "elo": checkpoint_elo(run, args.elo_deals, args.device, args.max_elo_ckpts),
    }
    (run / "report.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    (run / "report.md").write_text(to_markdown(data))
    print(to_markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
