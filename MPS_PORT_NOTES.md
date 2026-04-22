# Kimodo MPS Port Notes

This private package is a source-only Apple Silicon/MPS packaging of the locally
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

## Local Validation

The included manifest is `manifests/kimodo_mps_probe.json`.

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

## Practical MPS Notes

- The Kimodo denoiser itself is usable on MPS for short local probes.
- Full text-prompt generation also needs the local LLM2Vec text encoder or
  cached text embeddings.
- The default 8B text encoder is the main memory risk on a 24 GB Mac.
- For production use on Apple Silicon, prefer cached embeddings or low-frame,
  low-sample tests first, then increase duration after memory is confirmed.
- The bundled MotionCorrection extension is CMake/C++ based, not CUDA, but it
  was not built and end-to-end generation-tested in this packaging pass.

## Known Limits

- This is a pragmatic Apple Silicon compatibility port, not a full upstream
  support claim.
- Full interactive prompt generation with the 8B text encoder was not validated
  on this Mac.
- The validation path used the actual Kimodo checkpoint and MPS denoiser with
  controlled lightweight text conditioning for a tiny smoke test.
- Long, high-sample generations can still exceed unified memory.

## Files Intentionally Excluded

- Model weights and checkpoints.
- `.venv`, package caches, Hugging Face caches, and build artifacts.
- Generated NPZ/BVH/CSV/video/render outputs.
- Local workspace `.env` and any secrets.
