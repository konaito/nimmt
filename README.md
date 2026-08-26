# nimmt — 6 Nimmt! 強化学習AI

カードゲーム「ニムト（6 Nimmt!）」のAIを PPO 自己対戦＋リーグ学習で作るプロジェクト。
ブラウザで対戦できるWeb UI付き。

> 本プロジェクトは非公式のファン実装です。「6 Nimmt!」は Wolfgang Kramer 氏デザイン、
> AMIGO 社発売のカードゲームであり、本リポジトリはそれらと一切関係ありません。
> カードの絵柄等の公式アセットは含みません。

## 強さ（実測）

最強モデル `runs/exp4/latest.pt`（4人戦）の対ベースライン成績。2万ディールの重複配牌
（全席ローテーション、計8万対戦）、95%CIはペア差ブートストラップ（n=10000）:

| 相手 | AI平均失点/ディール | 相手平均失点 | 差 | 95%CI |
|---|---|---|---|---|
| random | 5.74 | 15.76 | -10.02 | [-10.07, -9.96] |
| greedy（最小牛頭行に貪欲） | 9.88 | 13.57 | -3.70 | [-3.76, -3.63] |
| heuristic（安全マージン考慮） | 9.81 | 13.10 | -3.29 | [-3.36, -3.23] |
| mc-rollout（1手96プレイアウトの先読み） | 10.06 | 12.46 | -2.40 | [-2.55, -2.24] |

先読みなしの1回のforward（約24万パラメータのMLP、931KB）でこの数字。
詳細は [docs/results.md](docs/results.md)。

先行研究 [johannbrehmer/rl-6-nimmt](https://github.com/johannbrehmer/rl-6-nimmt) の
AlphaZero風MCTS（Alpha0.5, 800プレイアウト/手）とも相手の環境で直接対戦し、
差 -1.92牛頭/ディール（95%CI [-2.80, -1.02]）で有意勝ち。方法と全数字は
[docs/vs-alpha05.md](docs/vs-alpha05.md)。

## 遊ぶ

**ブラウザですぐ遊ぶ（インストール不要）**: https://konaito.github.io/nimmt/

ローカルで動かす場合:

```bash
uv sync --group web
uv run --group web uvicorn webapp.server:app --port 8765
# → http://localhost:8765 を開く
```

2人戦（vs exp2）と4人戦（vs exp4）を選べる66点マッチ。CLI版もある:

```bash
uv run python scripts/play.py --checkpoint runs/exp4/latest.pt --players 4
```

## 学習を再現する

```bash
uv sync
uv run pytest -q                                  # ルール正当性テスト
uv run python scripts/crosscheck.py 100000 4      # 参照実装と10万ディール照合

# 本命レシピ（exp3 → exp4 の2段階、CPUで各1.5〜2時間目安）
uv run python -m nimmt.rl.train --run-dir runs/exp3 --players 4 \
  --iterations 3000 --reward-mode relative --eval-every 100
uv run python -m nimmt.rl.train --run-dir runs/exp4 --players 4 \
  --iterations 3000 --reward-mode relative --init-from runs/exp3/latest.pt \
  --ent-start 0.002 --ent-end 0.002 --seed 1 --eval-every 100

# 評価
uv run python scripts/report.py --run-dir runs/exp4 --deals 20000
uv run python scripts/ablation.py --run-a runs/exp4 --run-b runs/exp3 --deals 20000
```

## 構成

```
src/nimmt/
  cards.py       カード・牛頭の定義
  reference.py   参照実装（正しさの基準）
  vec.py         ベクトル化環境 VecNimmt（numpy、数千ゲーム並列）
  features.py    観測エンコーダ（266次元）
  arena.py elo.py  評価ハーネス（重複配牌・ペア差CI・Bradley-Terry）
  bots/          random / greedy / heuristic / mc-rollout / neural
  rl/            PPO・リーグ・学習エントリポイント（train.py）
webapp/          FastAPI + 素のHTML/JS の対戦UI
scripts/         play / report / ablation / crosscheck / bench
docs/            評価結果・設計記録
```

設計のポイント:

- 環境はルールを2回実装（素直な参照実装と高速なベクトル化実装）し、10万ディールの
  完全一致で正しさを担保
- 評価は重複配牌＋ペア差ブートストラップCIで運の分散を消す
- モデルは共有エンコーダ＋カード/行/価値の3ヘッド。手札は per-card 特徴でスコアリング
  するので手札枚数に非依存
- 報酬は順位ベース（relative）が絶対失点（absolute）より強かった（1シード対での実測）

## License

MIT
