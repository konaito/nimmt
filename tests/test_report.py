import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_eval_seed_differs_from_the_training_monitor_seed():
    """最終レポートの配牌は、学習中のモニタリングと別のシードで引くこと。

    同じ配牌で見出し数字を出すと、モデル選択に使った盤面で自己採点することになる。
    """
    import report
    from nimmt.rl.train import MONITOR_SEED

    assert report.EVAL_SEED != MONITOR_SEED


def test_checkpoint_elo_is_symmetric_for_identical_nets(tmp_path):
    """同じ重みのチェックポイント同士なら Elo 差は 0 になること（席の偏りが無いことの確認）。

    `[i, j, i, j]` の並びと run_match の席ローテーションで偏りが相殺されているかを見る。
    """
    import torch

    from nimmt.rl.model import NimmtNet, save
    from report import checkpoint_elo

    torch.manual_seed(0)
    net = NimmtNet(hidden=16, depth=1)
    run = tmp_path / "run"
    run.mkdir()
    save(net, run / "ckpt_000100.pt")
    save(net, run / "ckpt_000200.pt")

    elo = checkpoint_elo(run, deals=60, device="cpu")
    assert len(elo["ratings"]) == 2
    assert abs(elo["ratings"][0] - elo["ratings"][1]) < 1e-6


def test_only_mc_rollout_uses_the_reduced_deal_count(tmp_path):
    """ディール数の絞り込みが mc-rollout だけに効き、他のベースラインには波及しないこと。"""
    import torch

    from nimmt.rl.model import NimmtNet
    from report import vs_baselines

    torch.manual_seed(0)
    net = NimmtNet(hidden=16, depth=1)
    rows = vs_baselines(net, 4, deals=24, device="cpu", skip_rollout=False,
                        rollout_deals=8, rollout_samples=2)
    by = {r["opponent"]: r["n_deals"] for r in rows}
    assert by["random"] == by["greedy"] == by["heuristic"] == 24 * 4
    assert by["mc-rollout"] == 8 * 4


def test_report_runs_on_tiny_settings(tmp_path):
    import torch

    from nimmt.rl.model import NimmtNet, save
    from report import main

    run = tmp_path / "run"
    run.mkdir()
    torch.manual_seed(0)
    for it in (100, 200):
        save(NimmtNet(hidden=16, depth=1), run / f"ckpt_{it:06d}.pt")
    save(NimmtNet(hidden=16, depth=1), run / "latest.pt")

    rc = main(["--run-dir", str(run), "--deals", "100", "--elo-deals", "50",
               "--skip-rollout"])
    assert rc == 0
    data = json.loads((run / "report.json").read_text())
    assert "baselines_4p" in data and "baselines_2p" in data and "elo" in data
    for entry in data["baselines_4p"]:
        assert {"opponent", "neural_mean_loss", "opp_mean_loss", "diff", "ci_low", "ci_high",
                "neural_win_rate", "n_deals"} <= set(entry)
    md = (run / "report.md").read_text()
    assert "heuristic" in md and "95%CI" in md
