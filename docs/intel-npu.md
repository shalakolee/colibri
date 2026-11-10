# Intel NPU experiment (Windows, Intel Core Ultra)

Colibri currently has **no Intel NPU inference backend**. The AMD XDNA2
integration described in [xdna.md](xdna.md) is not compatible with Intel AI
Boost. Do not pass `--xdna` on an Intel laptop expecting NPU acceleration.

This branch adds a standalone **OpenVINO NPU qualification probe**, not an
inference integration. It exercises an explicit `NPU` matmul with dimensions
matching the GLM shared-expert gate/up projection. It never selects `AUTO`,
`HETERO`, or `MULTI`, so a successful run is not a silent CPU fallback.

## Run on the laptop

Install/update the Intel NPU driver from HP/Intel for your exact device, then:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade openvino numpy
python c/tools/probe_intel_npu.py
python c/tools/probe_intel_npu.py --m 64 --iterations 20
```

The probe prints available OpenVINO devices, NPU compile status, timing and
maximum error relative to a float32 NumPy reference. Failure is a non-zero
exit status, with no fallback.

## Required for a real Colibri NPU backend

1. Define a tightly scoped operation (e.g. GLM shared-expert gate/up) and
   exact tensor layout, quantization, weight lifetime and shapes.
2. Convert the stored grouped-int4 Colibri weights to a supported OpenVINO
   input or compile an equivalent quantized graph; do not assume that
   `fmt=4` can be loaded as-is.
3. Add a Windows-only optional helper/ABI with lazily loaded dependencies,
   model validation, persistent compiled models and weight caching.
4. Route the eligible operation into the helper, leaving all other operations
   on existing CPU/Vulkan paths. Failed operations must not publish partial
   output and must use the original path.
5. Validate numerical differences, end-to-end prompt/decode throughput,
   memory overhead, actual NPU activity and correctness on physical hardware.

This tool only completes the **hardware and synthetic-kernel feasibility**
portion of that plan. It does not accelerate `coli chat`, `coli serve` or
model inference. NPU TOPS ratings are not token/s benchmarks.
