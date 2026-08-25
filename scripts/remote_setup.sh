#!/usr/bin/env bash
# リモートに uv と Python 3.12 を入れ、作業ディレクトリを作る。冪等。
set -euo pipefail

HOST="${NIMMT_HOST:-USER@REMOTE_HOST}"
REMOTE_DIR="${NIMMT_REMOTE_DIR:-nimmt}"

echo "== 接続確認 =="
ssh -o ConnectTimeout=10 "$HOST" 'sw_vers -productVersion; sysctl -n machdep.cpu.brand_string; sysctl -n hw.ncpu'

echo "== uv =="
ssh "$HOST" 'command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh'

echo "== Python 3.12 =="
ssh "$HOST" 'export PATH="$HOME/.local/bin:$PATH"; uv --version; uv python install 3.12'

echo "== 作業ディレクトリ =="
ssh "$HOST" "mkdir -p ~/$REMOTE_DIR"

echo "== 常駐実行の手段 =="
# tmux はリモートに入っていない（実測）。ユーザーのマシンなので勝手に brew install はしない。
# nohup + caffeinate で代替する。caffeinate は設定を変えずプロセスが生きている間だけスリープを抑止する。
ssh "$HOST" 'command -v caffeinate >/dev/null && echo "caffeinate OK" || echo "caffeinate が無い"'

echo "OK"
