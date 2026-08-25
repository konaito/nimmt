# リモート学習環境（M4 Mac mini, USER@REMOTE_HOST）

Task 12 で構築した、学習を回すためのリモート環境の記録。

## リモートのスペック（実測値）

`ssh USER@REMOTE_HOST` で以下を実行して取得（2026-08-24実測）。

```
$ sw_vers
ProductName:    macOS
ProductVersion: 26.4
BuildVersion:   25E246

$ sysctl -n machdep.cpu.brand_string
Apple M4

$ sysctl -n hw.ncpu
10

$ sysctl -n hw.memsize
17179869184   # = 16 GiB
```

- OS: macOS 26.4 (build 25E246)
- CPU: Apple M4 / 10コア
- メモリ: 16 GiB
- Python: システムは 3.9.6 のみ。`uv python install 3.12` で 3.12.14 を uv 管理下に追加済み（システムの 3.9.6 には触っていない）
- uv: 未インストールだったのでインストール済み（`~/.local/bin`、公式インストーラ経由）。バージョン 0.12.5
- tmux: 未インストール（実測）。ユーザーの実マシンのため `brew install tmux` はしない。代わりに `nohup` + `caffeinate` で常駐実行する（下記「学習の起動コマンド」参照）。`caffeinate` は macOS 標準コマンドで、リモートに実在することを実測確認済み（`caffeinate OK`）。

## ベンチマーク結果（`docs/bench-remote.json`）

`scripts/bench.py` を **3回** 実行し、`run1`〜`run3` として全実測値を `docs/bench-remote.json` に残した（代表1回ではなく3回分すべてを保存する方式を選択）。各回は内部で `bench_iteration` が `reps=5`（先頭1回はウォームアップとして破棄）を回し、中央値・全実測値・loadavg を記録している。

### `bench_iteration`（games=2048, players=4, hidden=256, depth=2）の中央値

| run | device | collect_sec (median) | update_sec (median) | total_sec (median) | games/sec (median) |
|---|---|---|---|---|---|
| 1 | cpu | 0.3088 | 2.0715 | 2.3802 | 860.4 |
| 1 | mps | 0.2411 | 2.2144 | 2.4755 | 827.3 |
| 2 | cpu | 0.3116 | 2.0713 | 2.3829 | 859.5 |
| 2 | mps | 0.2395 | 2.1677 | 2.4072 | 850.8 |
| 3 | cpu | 0.3127 | 2.0742 | 2.3870 | 858.0 |
| 3 | mps | 0.2419 | 2.1396 | 2.3815 | 860.0 |

**total_sec_median での CPU/MPS の勝敗は安定しない**（run1: CPU勝ち約4%差、run2: CPU勝ち約1%差、run3: MPS勝ち約0.2%差）。差はどの回も0〜4%とノイズレベルで、3回中2回CPU、1回MPSという結果だった。無理に「MPSが速い／CPUが速い」と結論づけられる差ではない。

ただし内訳を見ると、**collect（自己対戦の経験収集）と update（PPOの勾配更新）の勝敗はそれぞれ3回とも安定している**。

- collect_sec: **3回ともMPSが約22〜23%速い**（0.24s台 vs 0.31s台）
- update_sec: **3回ともCPUが約3〜7%速い**（2.07s台 vs 2.14〜2.21s台）

update が total の約87%を占める（2.07s / 2.38s）ため、update側の優位が総合時間を支配する。これが total_sec_median の勝敗が2/3でCPU寄りになった理由と整合する。

### 参考: `bench_net`（forward/backward 単体、hidden=256, n=2048）

```
cpu:  fwd 3.9ms / bwd 11.1ms
mps:  fwd 7.2ms / bwd 20.1ms
```
n=2048（1イテレーションの実データ規模に近い）では forward/backward 単体でも CPU の方が速い。n が大きくなる（8192, 32768）につれて MPS が forward で追いつく傾向はあるが、今回の学習設定（games=2048）のスケールでは CPU 優位という上記の結果と整合的。

### `bench_env`（VecNimmt + HeuristicBot のスループット）

batch=2048 で約31,000 deals/sec、約133万 decisions/sec。CPU/MPS は関係ない（env はCPU上のnumpy実装）。

### 決めた設定と根拠

- **`--device cpu`**
  根拠: `total_sec_median` 単体では3回中2回CPU勝ち・1回MPS勝ちで「安定して速い」とは言えない（差0〜4%）。しかし内訳を見ると `update_sec`（total の約87%を占める）は3回とも安定してCPUが3〜7%速く、`collect_sec`（約13%）は3回とも安定してMPSが22〜23%速い。重みの大きいupdate側の優位を採ってCPUを選んだ。**これは決定的な性能差ではなく、内訳の安定パターンに基づく弱い根拠であることを明記する。** total_sec だけを見るなら「決められない」が正直な結論。
- **`--games 2048`**
  brief記載の起動コマンド例と、ベンチで実測した規模（games=2048）を一致させた。この規模で CPU なら1イテレーション中央値 約2.38秒（3回の中央値の平均）。3000イテレーションなら概算 2.38s×3000 ≈ 7140秒 ≈ 約2時間（実測値からの単純外挿であり保証ではない）。
- **`--hidden 256`**
  ベンチで使った値をそのまま採用。`bench_net` でも hidden=256 でCPU/MPSとも実用的な速度が出ており、変更する根拠となる実測結果は得ていない。

## 学習の起動コマンド（`nohup` + `caffeinate`。tmux はリモートに入っていないので使わない）

```bash
ssh USER@REMOTE_HOST 'export PATH="$HOME/.local/bin:$PATH"; cd ~/nimmt && \
  mkdir -p runs/exp1 && nohup caffeinate -is uv run python -m nimmt.rl.train \
    --run-dir runs/exp1 --iterations 3000 --games 2048 --players 4 \
    --hidden 256 --depth 2 --device cpu --eval-every 100 \
    > runs/exp1/stdout.log 2>&1 < /dev/null & echo "pid=$!"'
```

`caffeinate -is` は学習プロセスが生きている間だけスリープを抑止する（`-i` アイドルスリープ抑止、`-s` システムスリープ抑止）。システム設定は変更しないので、プロセスが終われば元の設定に自動で戻る。
`nohup ... < /dev/null &` にしているので、ssh セッションを切っても学習は走り続ける。

## ログ回収の手順

```bash
# コード変更をリモートへ送る
./scripts/sync.sh

# まだ学習プロセスが生きているか
ssh USER@REMOTE_HOST 'pgrep -fl "nimmt.rl.train" || echo "プロセスなし"'

# 進捗確認（ログの最新3行を tail）
ssh USER@REMOTE_HOST 'tail -3 ~/nimmt/runs/exp1/log.jsonl'

# 学習結果（runs/ 以下）をローカルへ回収
./scripts/sync.sh --fetch
```

`sync.sh --fetch` は `~/nimmt/runs/` をローカルの `runs/` にミラーする（`.venv` や `__pycache__` は送信側の除外対象。fetch は `runs/` のみ対象）。`--fetch` 方向も `runs/remote-smoke/` を対象に実測動作確認済み（`config.json` / `latest.pt` / `log.jsonl` の3ファイルを取得し、`log.jsonl` の内容がローカル版と非タイミング項目で完全一致することを確認した）。

## 途中経過の見方

`runs/exp1/log.jsonl` は1イテレーション1行のJSONL。各行に含まれる主なキー:

- `iter`: イテレーション番号
- `mean_loss`: そのイテレーションの平均損失（対戦の平均失点。低いほど良い）
- `sec` / `games_per_sec`: そのイテレーションの所要時間とスループット
- `policy_loss` / `value_loss` / `entropy` / `clip_frac` / `approx_kl`: PPOの内部指標
- `eval_every`（`--eval-every` の倍数のイテレーションのみ）で以下が追加される
  - `eval_heuristic` / `eval_greedy` / `eval_random`: 各ボットと対戦させたときのneuralの平均失点
  - `eval_*_opp`: 相手側の平均失点
  - `eval_*_winrate`: neuralの勝率（4人戦なので基準は0.25）

`eval_heuristic_winrate` が 0.25 を上回っていくかが、heuristic bot に対して学習が進んでいるかの目安になる。

進捗を素早く見るには:

```bash
ssh USER@REMOTE_HOST 'tail -3 ~/nimmt/runs/exp1/log.jsonl'
```

`latest.pt` は毎イテレーション上書きされる最新チェックポイント、`ckpt_XXXXXX.pt` は `--ckpt-every` ごとのスナップショット。

## ローカルでの事前検証（Step 3, リモートに投げる前に実施）

```
$ uv run python -m nimmt.rl.train --run-dir runs/local-smoke --iterations 5 \
    --games 256 --minibatch 2048 --eval-every 5 --eval-deals 200
（5イテレーション完走、最終行に eval_heuristic 等を含む）
$ ls runs/local-smoke/
config.json  latest.pt  log.jsonl
$ cat runs/local-smoke/log.jsonl | wc -l
5
```

同じコマンドをリモートでも実行し（`runs/remote-smoke/log.jsonl`）、seed固定のため `mean_loss` / `eval_heuristic` 等の学習結果は全イテレーションでローカルと完全一致することを確認した（`diff` で確認。`sec` / `games_per_sec` はマシン依存の実測値なので当然異なる）。

## 既知の制約・逸脱

- `scripts/sync.sh` は brief 記載のスクリプトから `rsync` のフラグを1点変更した。brief の `--info=progress2` はGNU rsync専用オプションで、ローカル（macOS標準の openrsync, protocol 29）では `unrecognized option` で失敗することをローカル検証で発見した。openrsync/GNU rsync 両対応の `--progress` に置き換えた。動作はローカル・リモートの両方の sync（push / `--fetch` 双方向）で確認済み。
- `scripts/sync.sh` の除外リストに `.superpowers` を追加した。計画・レビュー用の内部成果物がリモートに複製されるのを防ぐため（宛先は本人のマシンなので実害はないが、ノイズになる）。
- リモートに `tmux` は無いが、これは解決済み。学習の常駐実行は `nohup` + `caffeinate` で行う（上記「学習の起動コマンド」参照）。`brew install tmux` のような許可範囲外の操作をユーザーに要求する必要はない。
