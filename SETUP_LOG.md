# Environment Setup Log

**Date:** 2026-03-29
**Machine:** Linux, 5x NVIDIA H200 (143771 MiB each), CUDA 12.9

## Steps Executed

### 1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Installed uv 0.11.2 to /root/.local/bin
```

### 2. Create Python 3.11 virtual environment
```bash
uv venv --python 3.11 /workspace/dreamzero/.venv
# Downloaded cpython-3.11.15, created .venv
```

### 3. Install project dependencies
```bash
uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
# Installed 262 packages including:
#   torch==2.8.0+cu129
#   torchvision==0.23.0+cu129
#   torchaudio==2.8.0+cu129
#   transformers==4.51.3
#   deepspeed==0.18.8
#   diffusers==0.30.2
#   peft==0.5.0
#   datasets==3.6.0
#   ray==2.47.1
```
> `--index-strategy unsafe-best-match` was required because `requests` was pinned to an old version on the PyTorch index, blocking `datasets==3.6.0`.

### 4. Install `flash-attn`
```bash
MAX_JOBS=8 uv pip install --no-build-isolation flash-attn
# Built and installed flash-attn==2.8.3
```

> GB200-only steps (Transformer Engine, TensorRT) were skipped — not needed for H200.

### 5. Fix `libGL.so.1` missing (headless server)
`opencv-python` requires `libGL.so.1` which is absent on headless servers. Replace it with the headless variant pinned to the project version:
```bash
uv pip uninstall opencv-python
uv pip install opencv-python-headless==4.8.0.74
# numpy gets bumped as a side effect — pin it back
uv pip install numpy==1.26.4
```
> `opencv-python-headless` is already resolved by the install in step 3 but shadowed by `opencv-python`. Removing the non-headless package fixes the `ImportError: libGL.so.1` crash at startup.
> Pinning to `4.8.0.74` (the project's version) is important — `4.11` installs with an incomplete native extension (missing `.so`), causing `AttributeError: module 'cv2' has no attribute 'CV_8U'`.

### 6. Download pretrained checkpoint
```bash
mkdir -p checkpoints
.venv/bin/huggingface-cli download GEAR-Dreams/DreamZero-DROID \
  --repo-type model \
  --local-dir ./checkpoints/DreamZero-DROID
```

## Result

| Component | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.8.0+cu129 |
| torchvision | 0.23.0+cu129 |
| torchaudio | 2.8.0+cu129 |
| flash-attn | 2.8.3 |
| CUDA | 12.9 |

## Activating the Environment

```bash
source /workspace/dreamzero/.venv/bin/activate
```
