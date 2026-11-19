"""Diagnostic TSF capture through the experimental private Kineto AON adapter."""

import argparse
import ctypes
import json
import time
from pathlib import Path

import torch


def load_adapter():
    library = Path(torch._C.__file__).parent / "lib" / "libkineto_aon.so"
    if not library.is_file():
        raise RuntimeError(f"Build PyTorch with KINETO_ENABLE_AON=ON: {library}")
    adapter = ctypes.CDLL(str(library))
    adapter.kineto_aon_service_available.argtypes = []
    adapter.kineto_aon_service_available.restype = ctypes.c_int
    adapter.kineto_aon_consumer_create.argtypes = []
    adapter.kineto_aon_consumer_create.restype = ctypes.c_void_p
    adapter.kineto_aon_consumer_initialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    adapter.kineto_aon_consumer_initialize.restype = ctypes.c_int
    for name in ("start", "stop"):
        function = getattr(adapter, f"kineto_aon_consumer_{name}")
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
    adapter.kineto_aon_consumer_destroy.argtypes = [ctypes.c_void_p]
    adapter.kineto_aon_consumer_destroy.restype = None
    return adapter


def capture(adapter, database: Path, seconds: float) -> dict:
    if database.exists():
        raise FileExistsError(f"Choose a new capture directory: {database}")
    consumer = adapter.kineto_aon_consumer_create()
    if not consumer:
        raise RuntimeError("Cannot allocate the AON streaming consumer")
    iterations = 0
    try:
        if not adapter.kineto_aon_consumer_initialize(consumer, bytes(database)):
            raise RuntimeError(
                "Cannot initialize the service connection or trace storage"
            )
        if not adapter.kineto_aon_consumer_start(consumer):
            raise RuntimeError("Cannot start the AON streaming consumer")
        try:
            x = torch.randn(1024, 1024, device="cuda")
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                torch.mm(x, x)
                torch.cuda.synchronize()
                iterations += 1
        finally:
            if not adapter.kineto_aon_consumer_stop(consumer):
                raise RuntimeError("Cannot stop and flush the AON streaming consumer")
    finally:
        adapter.kineto_aon_consumer_destroy(consumer)
    blocks = list(database.glob("*.nvtsf"))
    if not blocks:
        raise RuntimeError(f"No TSF data blocks were written to {database}")
    return {
        "database": str(database),
        "iterations": iterations,
        "tsf_blocks": len(blocks),
        "allocated_bytes": sum(block.stat().st_size for block in blocks),
        "validation": "Inspect the TSF with SDK tools to verify API and GPU records",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--captures", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=2)
    args = parser.parse_args()
    if args.captures < 1 or not 0 < args.seconds < float("inf"):
        parser.error("--captures and --seconds must be positive and finite")
    adapter = load_adapter()
    if not adapter.kineto_aon_service_available():
        raise RuntimeError(
            "Start the external service and check XDG_RUNTIME_DIR and permissions"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = [
        capture(adapter, output / f"capture-{index}", args.seconds)
        for index in range(args.captures)
    ]
    (output / "captures.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
