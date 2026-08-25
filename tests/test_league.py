import json

import numpy as np
import torch

from nimmt.rl.league import CheckpointPool
from nimmt.rl.model import NimmtNet
from nimmt.rl.rollout import REWARD_SCALE, collect_selfplay
from nimmt.rl.buffer import TrajectoryBuffer
from nimmt.vec import VecNimmt


def test_pool_holds_independent_copies():
    pool = CheckpointPool(max_size=2)
    net = NimmtNet(hidden=16, depth=1)
    pool.add(net)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    pool.add(net)
    assert len(pool) == 2
    a = pool.sample(np.random.default_rng(0), net)
    assert all(not p.requires_grad for p in a.parameters()), "対戦相手は勾配を持たない"
    assert not a.training, "対戦相手は eval モード"
    pool.add(net)
    assert len(pool) == 2, "FIFO で上限を守ること"


def test_relative_reward_sums_to_zero_across_seats():
    net = NimmtNet(hidden=16, depth=1)
    env = VecNimmt(32, 4, seed=2)
    buf = TrajectoryBuffer(n_keys=32 * 4)
    collect_selfplay(net, env, buf, rng=np.random.default_rng(0), reward_mode="relative")
    b = buf.finish(lam=1.0, normalize=False)
    # 相対報酬なら、1ディールの全席のリターン合計は 0
    first = b.card_ret[: 32 * 4].reshape(32, 4)
    assert np.allclose(first.sum(axis=1), 0.0, atol=1e-4)


def test_train_smoke_writes_logs_and_checkpoint(tmp_path):
    from nimmt.rl.train import main

    rc = main([
        "--iterations", "3", "--games", "64", "--players", "4",
        "--hidden", "32", "--depth", "1", "--minibatch", "1024",
        "--eval-every", "2", "--eval-deals", "100",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert rc == 0
    run = tmp_path / "run"
    assert (run / "config.json").exists()
    assert (run / "latest.pt").exists()
    lines = (run / "log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    for k in ("iter", "mean_loss", "entropy", "sec", "games_per_sec"):
        assert k in rec
    assert any("eval_heuristic" in json.loads(l) for l in lines)
