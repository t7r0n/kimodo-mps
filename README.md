# Kimodo MPS for Apple Silicon

This repository is a Mac/Apple Silicon compatibility package for NVIDIA
Kimodo. It keeps the upstream Kimodo source, adds MPS-focused fixes, and
documents the exact local validation used to prove that a text prompt can
generate motion on this Mac.

The original NVIDIA project remains the authority for Kimodo model design,
licenses, datasets, benchmark context, and general Linux/CUDA usage:

- Original repo: https://github.com/nv-tlabs/kimodo
- Project page: https://research.nvidia.com/labs/sil/projects/kimodo/
- Upstream README preserved here: [UPSTREAM_README.md](UPSTREAM_README.md)
- Detailed Mac port notes: [MPS_PORT_NOTES.md](MPS_PORT_NOTES.md)

This repo is source-only. It does not include model weights, Hugging Face
caches, `.env` files, virtual environments, generated motions, or local secrets.

## Current Status

Validated on Apple Silicon with PyTorch MPS:

- Full CLI text prompt path works with the official local LLM2Vec/Llama-3 text
  encoder.
- Kimodo-SOMA-RP-v1.1 loads and runs on `mps:0`.
- The smoke test generates both `.npz` and `.bvh`.
- The BVH export MPS float64 issue is fixed in this package.

The validated smoke test was intentionally tiny:

- Prompt: `A person walks forward.`
- Model: `kimodo-soma-rp`, resolved as `Kimodo-SOMA-RP-v1.1`
- Duration: `0.25` seconds
- Frames: `7`
- Diffusion steps: `1`
- Output: NPZ plus BVH

Validated files from the local run:

- NPZ: 47,392 bytes
- BVH: 30,217 bytes
- BVH frame time: `0.03333333333333333`
- NPZ arrays: `foot_contacts`, `global_root_heading`, `global_rot_mats`,
  `local_rot_mats`, `posed_joints`, `root_positions`, `smooth_root_pos`

Manifest: [manifests/kimodo_mps_e2e_prompt_probe.json](manifests/kimodo_mps_e2e_prompt_probe.json)

## What Changed for Mac/MPS

- CLI, demo, benchmark, and helper entrypoints now prefer devices in this order:
  `cuda`, then `mps`, then `cpu`.
- The Kimodo model switches to `float32` on MPS because Apple MPS does not
  support every CUDA-oriented half precision path used by the upstream code.
- `LLM2VecEncoder(device="auto")` now resolves to MPS on Apple Silicon.
- CUDA-only seed and deterministic setup calls are guarded so they do not crash
  on a Mac with no CUDA.
- The demo health check no longer treats MPS as CUDA.
- Benchmark scripts use the same local device selection logic.
- BVH export is forced to CPU during MPS runs because SOMA skeleton assets can
  contain `float64` buffers, and MPS cannot represent `float64`.
- Skeleton rotation conversion now creates identity matrices on the active
  tensor device and dtype instead of always creating CPU float32 tensors.
- Added `scripts/mps_e2e_smoke.sh` to run the Mac validation command.
- Added validation manifests under [manifests](manifests).

## Why the BVH Export Fix Matters

The first full `--bvh` run generated the Kimodo motion successfully but failed
while exporting BVH:

```text
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.
```

The failure came from moving SOMA skeleton buffers to MPS for plain-text BVH
serialization. BVH writing does not need GPU execution, so this package keeps
that export path on CPU for MPS runs and casts motion tensors to float32. After
that fix, the same prompt generated both:

- `walk_forward_bvh.npz`
- `walk_forward_bvh.bvh`

## Requirements

- Apple Silicon Mac.
- Python 3.11+ recommended.
- `uv` recommended.
- PyTorch with MPS support.
- Hugging Face token with access to:
  - `meta-llama/Meta-Llama-3-8B-Instruct`
  - `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
  - `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`
  - `nvidia/Kimodo-SOMA-RP-v1.1` or whichever Kimodo checkpoint you use

First-run local cache sizes observed during validation:

- Llama 3 8B text encoder: about 15 GB
- LLM2Vec MNTP adapter/cache: about 169 MB
- LLM2Vec supervised adapter/cache: about 160 MB
- Kimodo-SOMA-RP-v1.1 checkpoint: about 1.1 GB

The 8B text encoder is the main memory and disk risk on 24 GB unified-memory
Macs. Start with the smoke test before trying longer clips.

## Setup

```bash
git clone git@github.com:t7r0n/kimodo-mps.git
cd kimodo-mps

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

Authenticate with Hugging Face:

```bash
hf auth login
```

Or create a local `.env` file. The smoke script accepts common key formats:

```bash
HF_TOKEN=hf_xxx
```

## Run the MPS Smoke Test

```bash
scripts/mps_e2e_smoke.sh
```

The script runs:

```bash
TEXT_ENCODER_MODE=local \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
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
```

Expected outputs:

```text
outputs/mps_e2e_smoke/walk_forward.npz
outputs/mps_e2e_smoke/walk_forward.bvh
```

The first run can take a long time because the Llama-3 8B text encoder and
Kimodo checkpoint may need to download and load.

## Normal CLI Usage

After the smoke test works, increase duration and diffusion steps gradually:

```bash
TEXT_ENCODER_MODE=local \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m kimodo.scripts.generate \
  "A person walks slowly across the room." \
  --model kimodo-soma-rp \
  --duration 2.0 \
  --num_samples 1 \
  --diffusion_steps 25 \
  --seed 123 \
  --bvh \
  --output outputs/walk_slow
```

For production-quality motion, raise settings carefully and watch memory
pressure. Longer clips, multiple samples, more denoising steps, and
postprocessing can still exceed practical Mac limits.

## Known Limits

- The full CLI prompt-to-motion path is validated only as a tiny smoke test.
- The interactive Gradio demo was not separately validated on MPS.
- The bundled MotionCorrection extension is CMake/C++ based, but it was not
  built or validated in this Mac packaging pass.
- The upstream model still relies on a large LLM2Vec/Llama-3 text encoder.
- Cached embedding or no-network workflows would make Mac testing easier, but
  they are not implemented here yet.

## Licenses and Attribution

This package keeps upstream Kimodo licensing and attributions:

- [LICENSE](LICENSE)
- [ATTRIBUTIONS.MD](ATTRIBUTIONS.MD)
- [UPSTREAM_README.md](UPSTREAM_README.md)

Model checkpoints and gated text encoder weights are governed by their upstream
Hugging Face/NVIDIA/Meta model licenses and are not redistributed here.
