#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <r2-prefix> <destination-dir>" >&2
  exit 2
fi

R2_PREFIX=$1
DESTINATION_DIR=$2
R2_ENV_FILE=${R2_ENV_FILE:-"$HOME/.secrets/intrep-cloudflare-r2"}
RCLONE_TRANSFERS=${RCLONE_TRANSFERS:-8}
RCLONE_CHECKERS=${RCLONE_CHECKERS:-16}

if [[ ! -f "$R2_ENV_FILE" ]]; then
  echo "R2_ENV_FILE not found: $R2_ENV_FILE" >&2
  exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y rclone
  else
    echo "rclone is required" >&2
    exit 1
  fi
fi

set -a
. "$R2_ENV_FILE"
set +a

export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Other
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT"

mkdir -p "$DESTINATION_DIR"
rclone copy "r2:$R2_BUCKET/$R2_PREFIX" "$DESTINATION_DIR" \
  --s3-no-check-bucket \
  --transfers "$RCLONE_TRANSFERS" \
  --checkers "$RCLONE_CHECKERS" \
  --stats 30s \
  --stats-one-line
rclone size "r2:$R2_BUCKET/$R2_PREFIX" --json
