#!/usr/bin/env bash
# コードをリモートへ送る（既定）か、学習結果を回収する（--fetch）。
set -euo pipefail

HOST="${NIMMT_HOST:-USER@REMOTE_HOST}"
REMOTE_DIR="${NIMMT_REMOTE_DIR:-nimmt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXCLUDES=(--exclude '.venv' --exclude 'runs' --exclude '__pycache__'
          --exclude '.git' --exclude '.pytest_cache' --exclude '*.pyc'
          --exclude '.superpowers')

if [[ "${1:-}" == "--fetch" ]]; then
  mkdir -p "$HERE/runs"
  rsync -az --progress "$HOST:~/$REMOTE_DIR/runs/" "$HERE/runs/"
  echo "回収完了: $HERE/runs"
else
  rsync -az --delete "${EXCLUDES[@]}" --progress "$HERE/" "$HOST:~/$REMOTE_DIR/"
  echo "送信完了: $HOST:~/$REMOTE_DIR"
fi
