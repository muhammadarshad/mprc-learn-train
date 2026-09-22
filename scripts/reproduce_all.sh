#!/usr/bin/env bash
set -euo pipefail
python scripts/materialize_dataset.py
python scripts/test_W2_B1.py
python scripts/train_digits.py
python scripts/vit_vof_ablation.py
