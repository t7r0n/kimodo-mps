#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ".env" ] && [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN_FROM_ENV="$(python - <<'PY'
from pathlib import Path

env_path = Path(".env")
if not env_path.exists():
    raise SystemExit
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if "=" in line:
        key, value = line.split("=", 1)
    else:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
    if key.strip().lower() in {"hf_token", "hf-token", "huggingface_hub_token", "hugging_face_hub_token"}:
        print(value.strip().strip('"').strip("'"))
        break
PY
)"
  if [ -n "$HF_TOKEN_FROM_ENV" ]; then
    export HF_TOKEN="$HF_TOKEN_FROM_ENV"
  fi
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

mkdir -p outputs/mps_e2e_smoke

python -m kimodo.scripts.generate \
  "A person walks forward." \
  --model kimodo-soma-rp \
  --duration 0.25 \
  --num_samples 1 \
  --diffusion_steps 1 \
  --no-postprocess \
  --seed 123 \
  --bvh \
  --output outputs/mps_e2e_smoke/walk_forward
