#!/usr/bin/env bash
# scripts/download_data.sh
# ---------------------------------------------------------------------------
# Downloads the LendingClub dataset from Kaggle using the Kaggle CLI.
#
# Pre-requisites:
#   1. pip install kaggle
#   2. Place your Kaggle API token at ~/.kaggle/kaggle.json
#      (download from https://www.kaggle.com/settings → API → Create Token)
#
# Usage:
#   bash scripts/download_data.sh
# ---------------------------------------------------------------------------

set -euo pipefail

RAW_DIR="data/raw"
DATASET="wordsforthewise/lending-club"

echo "[download_data] Creating $RAW_DIR …"
mkdir -p "$RAW_DIR"

echo "[download_data] Downloading dataset: $DATASET"
kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip

echo ""
echo "[download_data] Files downloaded to $RAW_DIR:"
ls -lh "$RAW_DIR"

echo ""
echo "[download_data] Done.  Next step:"
echo "  bash scripts/run_pipeline.sh"
