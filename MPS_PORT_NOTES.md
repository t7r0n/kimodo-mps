# Kimodo MPS Port Notes

This package is a source-only Apple Silicon/MPS packaging of the locally
validated Kimodo motion-generation port. It is based on:

- `nv-tlabs/kimodo` at local source commit `301d3a1`.

Model checkpoints, local virtual environments, caches, generated motion outputs,
and workspace secrets are not included. Users must download checkpoints from
the upstream Hugging Face/NVIDIA sources under their respective model licenses.

## What Was Changed

- Common CLI, benchmark, and demo entrypoints now select `cuda` first, then
  `mps`, then `cpu`.
- `Kimodo` forces fp32 when running on MPS because Apple MPS does not support
  every fp16/bf16 path used by CUDA-oriented model code.
- `LLM2VecEncoder(device="auto")` now resolves to MPS on Apple Silicon instead
  of falling straight back to CPU.
- Seed utilities no longer call CUDA APIs when CUDA is unavailable.
- The demo health check no longer treats MPS like CUDA.
- Benchmark generation/evaluation helpers use the same local device selection.
- BVH export stays on CPU during MPS runs because SOMA skeleton buffers can
  contain float64 values, which MPS cannot represent.
- Skeleton rotation conversion now creates identity matrices on the active
  tensor device/dtype instead of always creating CPU float32 tensors.

## Local Validation

The included manifests are:

- `manifests/kimodo_mps_probe.json`
- `manifests/kimodo_mps_e2e_prompt_probe.json`

Validated on local Apple Silicon:

- Torch MPS available: true.
- Torch CUDA available: false.
- Resolved checkpoint family: `kimodo-soma-rp`.
- Loaded model device: `mps:0`.
- Loaded model dtype: `torch.float32`.
- Parameter count: `283.28M`.
- Tiny denoising/generation path produced finite MPS tensors:
  - `local_rot_mats`: `[8, 77, 3, 3]`
  - `global_rot_mats`: `[8, 77, 3, 3]`
  - `posed_joints`: `[8, 77, 3]`
  - `root_positions`: `[8, 3]`
  - `smooth_root_pos`: `[8, 3]`
  - `foot_contacts`: `[8, 6]`
  - `global_root_heading`: `[8, 2]`

Full text-prompt end-to-end validation also passed locally on MPS after Hugging
Face access was granted for `meta-llama/Meta-Llama-3-8B-Instruct`:

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

Validated output:

- NPZ: 47,392 bytes.
- BVH: 30,217 bytes.
- Frames: 7.
- Frame time: 0.03333333333333333.
- NPZ arrays: `foot_contacts`, `global_root_heading`, `global_rot_mats`,
  `local_rot_mats`, `posed_joints`, `root_positions`, and `smooth_root_pos`.

## Recommended Setup

Use Python 3.11+ on Apple Silicon:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

Then install any optional extras required by your target workflow, following the
upstream Kimodo documentation. Checkpoints are downloaded or resolved by the
upstream code paths and are intentionally not part of this repository.

To run the MPS smoke test:

```bash
scripts/mps_e2e_smoke.sh
```

The first run requires Hugging Face access to:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
- `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`
- the selected Kimodo checkpoint, such as `nvidia/Kimodo-SOMA-RP-v1.1`

## Practical MPS Notes

- The Kimodo denoiser and the full local LLM2Vec prompt path both validated on
  this Apple Silicon machine for a tiny smoke test.
- The default 8B text encoder remains the main memory and disk risk on a 24 GB
  Mac.
- For production use on Apple Silicon, prefer cached embeddings or low-frame,
  low-sample tests first, then increase duration after memory is confirmed.
- First-run local cache sizes were roughly 15 GB for Llama 3 8B, 169 MB for
  LLM2Vec-MNTP, 160 MB for LLM2Vec supervised, and 1.1 GB for the
  Kimodo-SOMA-RP-v1.1 checkpoint.
- The bundled MotionCorrection extension is CMake/C++ based, not CUDA, but it
  was not built and validated in this packaging pass.

## Known Limits

- This is a pragmatic Apple Silicon compatibility port, not a full upstream
  support claim.
- Full CLI text-prompt generation validated for a tiny smoke test. The
  interactive Gradio demo was not separately validated.
- Long, high-sample generations can still exceed unified memory.

## Files Intentionally Excluded

- Model weights and checkpoints.
- `.venv`, package caches, Hugging Face caches, and build artifacts.
- Generated NPZ/BVH/CSV/video/render outputs.
- Local workspace `.env` and any secrets.
