"""Export an AON-backed torch.profiler trace using an externally managed service."""

import argparse
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("aon-trace.json"))
    args = parser.parse_args()
    if os.environ.get("KINETO_GPU_PROFILER") != "aon":
        raise RuntimeError("Set KINETO_GPU_PROFILER=aon before starting Python")
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")
    if args.output.exists():
        raise FileExistsError(args.output)

    matrix = torch.randn(2048, 2048, device="cuda")
    for _ in range(5):
        matrix @ matrix
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(20):
            with torch.profiler.record_function("aon_matmul"):
                matrix @ matrix
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(args.output))
    trace = json.loads(args.output.read_text())
    kernels = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel" and event.get("dur", 0) > 0
    ]
    if not kernels:
        raise RuntimeError(
            f"No timed kernels in {args.output}; check service providers, driver compatibility, and calibration"
        )
    print(f"Wrote {len(kernels)} timed kernels to {args.output}")


if __name__ == "__main__":
    main()
