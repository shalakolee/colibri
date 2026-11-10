#!/usr/bin/env python3
"""Explicit Intel NPU/OpenVINO probe for Colibri backend development.

This is a hardware qualification tool, NOT an inference backend. It refuses CPU
fallback and measures a GLM shared-expert-shaped matrix multiplication on NPU.
"""
import argparse
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=1, help="logical token rows")
    p.add_argument("--k", type=int, default=6144, help="projection input width")
    p.add_argument("--n", type=int, default=2048, help="projection output width")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=10)
    args = p.parse_args()
    if min(args.m, args.k, args.n, args.iterations) <= 0 or args.warmup < 0:
        p.error("dimensions and iterations must be positive; warmup must be nonnegative")
    try:
        import numpy as np
        import openvino as ov
        from openvino import opset13 as ops
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install: pip install openvino numpy", file=sys.stderr)
        return 2

    core = ov.Core()
    devices = core.available_devices
    print(f"OpenVINO version: {ov.get_version()}")
    print(f"Available devices: {', '.join(devices) or '(none)'}")
    if "NPU" not in devices:
        print("No Intel NPU exposed to OpenVINO. Check Windows NPU drivers.", file=sys.stderr)
        return 3

    # Use f16 to match an NPU-capable floating-point workload. Colibri's
    # grouped-int4 format must be decoded/compiled separately in a real backend.
    rng = np.random.default_rng(42)
    a = rng.uniform(-0.05, 0.05, (args.m, args.k)).astype(np.float16)
    w = rng.uniform(-0.05, 0.05, (args.k, args.n)).astype(np.float16)
    param = ops.parameter([args.m, args.k], dtype=np.float16, name="activations")
    matmul = ops.matmul(param, ops.constant(w), False, False)
    model = ov.Model([matmul], [param], "colibri_intel_npu_matmul_probe")

    try:
        # Explicit NPU: never AUTO/MULTI/HETERO, which could quietly run on CPU.
        compiled = core.compile_model(model, "NPU")
        request = compiled.create_infer_request()
        for _ in range(args.warmup):
            request.infer({0: a})
        timings = []
        for _ in range(args.iterations):
            begin = time.perf_counter()
            result = request.infer({0: a})
            timings.append((time.perf_counter() - begin) * 1000)
        out = np.asarray(next(iter(result.values())), dtype=np.float32)
        reference = a.astype(np.float32) @ w.astype(np.float32)
        max_abs = float(np.max(np.abs(out - reference)))
        print(f"Compiled device: NPU (explicit)")
        print(f"Shape: ({args.m}, {args.k}) x ({args.k}, {args.n}); dtype: f16")
        print(f"Mean inference latency: {sum(timings)/len(timings):.2f} ms")
        print(f"Min inference latency: {min(timings):.2f} ms")
        print(f"Max absolute error vs float32 NumPy reference: {max_abs:.6g}")
        print("NOTE: compilation and accuracy for this synthetic matmul do not prove")
        print("      compatibility with Colibri's quantized weights or end-to-end speed.")
        return 0
    except Exception as exc:
        print(f"Explicit NPU compilation/inference failed: {exc}", file=sys.stderr)
        print("No CPU fallback was attempted.", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
