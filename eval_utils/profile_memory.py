"""
GPU memory profiler for DreamZero serving.

Measures:
  - Static model-weight memory per component (text_encoder, image_encoder, VAE, DiT)
  - Peak activation memory for each inference stage by wrapping sub-methods

Usage:
  python eval_utils/profile_memory.py --model_path ./checkpoints/dreamzero_droid_wan22_smoke

The script loads the model with the same code path as serve_dreamzero_wan22.py,
synthesises a dummy observation, and runs one full inference pass.
"""

import argparse
import functools
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

# Same dynamo limits as serve_dreamzero_wan22.py to avoid FailOnRecompileLimitHit
_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tianshou.data import Batch

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _mb(n_bytes: int) -> float:
    return n_bytes / 1024**2


def _sync():
    torch.cuda.synchronize()


def _mem_allocated() -> int:
    return torch.cuda.memory_allocated()


def _reset_peak():
    torch.cuda.reset_peak_memory_stats()


def _peak_allocated() -> int:
    return torch.cuda.max_memory_allocated()


# ---------------------------------------------------------------------------
# Method wrapper that records peak memory
# ---------------------------------------------------------------------------

class MemoryRecorder:
    """Collects per-stage memory stats."""
    def __init__(self):
        self.records: list[dict] = []

    def wrap(self, obj, method_name: str, stage_label: str, call_limit: int | None = None):
        """
        Replace obj.method_name with a wrapper that records memory stats.
        If call_limit is set, only the first N calls are recorded (rest are pass-through).
        """
        original = getattr(obj, method_name)
        call_count = [0]
        recorder = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            call_count[0] += 1
            n = call_count[0]
            label = f"{stage_label}" if call_limit is None or n == 1 else f"{stage_label}_call{n}"
            if call_limit is not None and n > call_limit:
                return original(*args, **kwargs)
            _sync()
            _reset_peak()
            baseline = _mem_allocated()
            result = original(*args, **kwargs)
            _sync()
            peak = _peak_allocated()
            after = _mem_allocated()
            recorder.records.append({
                "label": label,
                "baseline_MB": _mb(baseline),
                "peak_MB": _mb(peak),
                "peak_delta_MB": _mb(peak - baseline),
                "alloc_after_MB": _mb(after),
                "net_delta_MB": _mb(after - baseline),
            })
            return result

        setattr(obj, method_name, wrapper)

    def wrap_all_calls(self, obj, method_name: str, stage_label: str):
        """Wrap every call and record all of them."""
        self.wrap(obj, method_name, stage_label, call_limit=None)


def print_table(rows: list[dict]):
    if not rows:
        return
    cols = ["label", "baseline_MB", "peak_MB", "peak_delta_MB", "alloc_after_MB", "net_delta_MB"]
    headers = {
        "label": "Stage",
        "baseline_MB": "Baseline (MB)",
        "peak_MB": "Peak (MB)",
        "peak_delta_MB": "Peak Δ (MB)",
        "alloc_after_MB": "After (MB)",
        "net_delta_MB": "Net Δ (MB)",
    }
    widths = {
        c: max(
            len(headers[c]),
            max(len(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])) for r in rows),
        )
        for c in cols
    }

    def fmt_row(r):
        parts = []
        for c in cols:
            v = r[c]
            w = widths[c]
            parts.append(f"{v:<{w}}" if isinstance(v, str) else f"{v:>{w}.1f}")
        return "  ".join(parts)

    sep = "  ".join("-" * widths[c] for c in cols)
    print()
    print("  ".join(f"{headers[c]:<{widths[c]}}" for c in cols))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print()


# ---------------------------------------------------------------------------
# Static weight memory per component
# ---------------------------------------------------------------------------

def measure_weight_memory(action_head) -> dict[str, float]:
    """Sum parameter bytes per top-level child of the action head."""
    result: dict[str, int] = {}
    for name, module in action_head.named_children():
        b = sum(p.numel() * p.element_size() for p in module.parameters())
        b += sum(buf.numel() * buf.element_size() for buf in module.buffers())
        result[name] = b
    return {k: _mb(v) for k, v in sorted(result.items(), key=lambda x: -x[1])}


# ---------------------------------------------------------------------------
# KV-cache size
# ---------------------------------------------------------------------------

def _tensor_bytes(t) -> int:
    if isinstance(t, torch.Tensor):
        return t.numel() * t.element_size()
    if isinstance(t, (list, tuple)):
        return sum(_tensor_bytes(x) for x in t)
    return 0


def measure_kv_cache(action_head) -> float:
    total = 0
    for attr in ("kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg"):
        total += _tensor_bytes(getattr(action_head, attr, None))
    return _mb(total)


# ---------------------------------------------------------------------------
# Read expected resolution from checkpoint eval_transform
# ---------------------------------------------------------------------------

def _get_checkpoint_resolution(policy) -> tuple[int, int]:
    """Return (height, width) that the checkpoint's eval_transform expects."""
    from groot.vla.data.transform import ComposedModalityTransform
    eval_transform = getattr(policy, "eval_transform", None)
    if eval_transform is None or not isinstance(eval_transform, ComposedModalityTransform):
        return (180, 320)
    for t in eval_transform.transforms:
        if hasattr(t, "original_resolutions") and t.original_resolutions:
            w, h = next(iter(t.original_resolutions.values()))
            return (int(h), int(w))
    return (180, 320)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile GPU memory during DreamZero serving")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--embodiment_tag", default="oxe_droid")
    parser.add_argument("--image_height", type=int, default=160)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--num_frames", type=int, default=1,
                        help="Number of input frames (1 = first call / cache-cold)")
    args = parser.parse_args()

    # ---- Distributed init ----
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    # ---- Load model ----
    print(f"\n[profile_memory] Loading policy from {args.model_path} ...")
    _sync()
    mem_before_load = _mem_allocated()

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=args.model_path,
        device="cuda",
        device_mesh=device_mesh,
    )

    _sync()
    mem_after_load = _mem_allocated()
    total_model_MB = _mb(mem_after_load - mem_before_load)
    print(f"[profile_memory] Model loaded. GPU memory used: {_mb(mem_after_load):.1f} MB total, "
          f"{total_model_MB:.1f} MB for this model")

    # ---- Weight memory breakdown ----
    action_head = policy.trained_model.action_head
    weight_mem = measure_weight_memory(action_head)

    print("\n=== Weight memory by action-head component (parameters + buffers) ===")
    total_ah_w = sum(weight_mem.values())
    for comp, mb in weight_mem.items():
        print(f"  {comp:<30s}  {mb:>8.1f} MB  ({mb/1024:.3f} GB)")
    print(f"  {'TOTAL action_head':<30s}  {total_ah_w:>8.1f} MB  ({total_ah_w/1024:.3f} GB)")

    backbone = getattr(policy.trained_model, "backbone", None)
    if backbone is not None:
        bb_b = sum(p.numel() * p.element_size() for p in backbone.parameters())
        bb_b += sum(b.numel() * b.element_size() for b in backbone.buffers())
        print(f"  {'backbone (VLM)':<30s}  {_mb(bb_b):>8.1f} MB  ({_mb(bb_b)/1024:.3f} GB)")

    # ---- Instrument sub-methods ----
    recorder = MemoryRecorder()

    # Text encoder: encode_prompt is called once per CFG branch (positive + negative)
    recorder.wrap_all_calls(action_head, "encode_prompt", "text_encode")

    # Image encoder: encode_image is called once when current_start_frame == 0
    recorder.wrap_all_calls(action_head, "encode_image", "image_encode_clip")

    # VAE encode: called on every forward that isn't the first frame
    recorder.wrap_all_calls(action_head.vae, "encode", "vae_encode")

    # VAE decode: called at the end of every forward
    recorder.wrap_all_calls(action_head.vae, "decode", "vae_decode")

    # DiT forward: _run_diffusion_steps is called for prefill + each active diffusion step
    recorder.wrap_all_calls(action_head, "_run_diffusion_steps", "dit_forward")

    # ---- Dummy observation (use resolution from checkpoint eval_transform) ----
    h_cp, w_cp = _get_checkpoint_resolution(policy)
    H, W, T = h_cp, w_cp, args.num_frames
    print(f"[profile_memory] Using resolution {H}x{W} from checkpoint eval_transform")
    rng = np.random.default_rng(42)
    dummy_obs = {
        "video.exterior_image_1_left": rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
        "video.exterior_image_2_left": rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
        "video.wrist_image_left": rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
        "state.joint_position": rng.standard_normal((1, 7)).astype(np.float64),
        "state.gripper_position": rng.standard_normal((1, 1)).astype(np.float64),
        "annotation.language.action_text": "pick up the red cup",
    }

    # ---- Warmup (triggers torch.compile, shape specialisation, etc.) ----
    print("\n[profile_memory] Warmup inference (may take a while due to torch.compile) ...")
    with torch.no_grad():
        policy.lazy_joint_forward_causal(Batch(obs=dummy_obs.copy()))
    recorder.records.clear()

    # ---- Reset caches so we get a cold first-call profile ----
    action_head.current_start_frame = 0
    action_head.language = None
    action_head.kv_cache1 = None
    action_head.kv_cache_neg = None
    action_head.crossattn_cache = None
    action_head.crossattn_cache_neg = None
    action_head.clip_feas = None
    action_head.ys = None

    # ---- Profiled inference ----
    print("[profile_memory] Running profiled inference ...")
    _sync()
    _reset_peak()
    baseline_total = _mem_allocated()

    with torch.no_grad():
        policy.lazy_joint_forward_causal(Batch(obs=dummy_obs.copy()))

    _sync()
    peak_total = _peak_allocated()
    kv_cache_MB = measure_kv_cache(action_head)

    # ---- Aggregate DiT steps ----
    dit_records = [r for r in recorder.records if r["label"].startswith("dit_forward")]
    non_dit_records = [r for r in recorder.records if not r["label"].startswith("dit_forward")]

    if dit_records:
        # Separate KV-cache prefill (first _run_diffusion_steps call, action=None) from
        # the denoising loop calls. Prefill is always the first call.
        prefill = dit_records[0]
        prefill = dict(prefill); prefill["label"] = "dit_kv_prefill"
        denoise_records = dit_records[1:]

        aggregated = list(non_dit_records) + [prefill]
        if denoise_records:
            peak_deltas = [r["peak_delta_MB"] for r in denoise_records]
            aggregated.append({
                "label": f"dit_denoise_steps ({len(denoise_records)} calls, avg peak Δ)",
                "baseline_MB": denoise_records[0]["baseline_MB"],
                "peak_MB": max(r["peak_MB"] for r in denoise_records),
                "peak_delta_MB": sum(peak_deltas) / len(peak_deltas),
                "alloc_after_MB": denoise_records[-1]["alloc_after_MB"],
                "net_delta_MB": sum(r["net_delta_MB"] for r in denoise_records),
            })
    else:
        aggregated = list(recorder.records)

    # ---- Print results ----
    print(f"\n=== Total GPU memory after model load:   {_mb(mem_after_load):.1f} MB  ({_mb(mem_after_load)/1024:.3f} GB) ===")
    print(f"=== Baseline at inference start:         {_mb(baseline_total):.1f} MB ===")
    print(f"=== Peak GPU memory during inference:    {_mb(peak_total):.1f} MB  "
          f"(Δ {_mb(peak_total - baseline_total):.1f} MB above inference baseline) ===")
    print(f"=== KV cache tensors on GPU (after):     {kv_cache_MB:.1f} MB ===")

    print("\n=== Per-stage inference memory breakdown ===")
    print("  peak_delta_MB = max GPU mem above stage baseline (activation peak)")
    print("  net_delta_MB  = GPU memory retained after stage (e.g. output tensors, KV cache)")
    print_table(aggregated)

    if dit_records and len(dit_records) > 1:
        print("=== Individual DiT step breakdown ===")
        for i, r in enumerate(dit_records):
            label = "prefill" if i == 0 else f"step_{i-1:02d}"
            print(f"  {label:<10s}  peak_delta={r['peak_delta_MB']:>7.1f} MB  net_delta={r['net_delta_MB']:>7.1f} MB")
        print()


if __name__ == "__main__":
    main()
