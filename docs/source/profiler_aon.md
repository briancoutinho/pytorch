# Experimental NVIDIA AON client

NVIDIA Nsight Perf SDK for AI provides always-on (AON) GPU tracing through an
external service. This experimental Kineto integration compiles its public
NvTrace client into a private shared adapter, `libkineto_aon.so`.

```{warning}
This backend is experimental. Setting `KINETO_GPU_PROFILER=aon` selects
AON instead of CUPTI for GPU collection. It requires a separately running,
compatible service that captures CUDA API and GPU timing records. PM samples
alone cannot produce CUDA activity events. Validate service readiness and the
event categories in each capture; a successful connection alone does not
establish that every provider works.
```

The profiler example below exports Kineto activities. The raw-capture examples
also test the private adapter directly and write an SDK Trace Stream Database
(TSF). This diagnostic interface is not a supported public
PyTorch Python API and may change. A nonempty TSF file alone does not prove
that CUDA API records or timed kernels were captured.

## Requirements

- Build PyTorch from source with CUDA, Kineto, and `KINETO_ENABLE_AON=ON` on Linux.
- Use a supported NVIDIA GPU and driver configuration. Consult the
  [SDK prerequisites](https://github.com/NVIDIA/nsight-perf-sdk-for-ai#prerequisites)
  for the current hardware, operating-system, CUDA, and driver requirements.
- Install the [Nsight Perf service and tools](https://github.com/NVIDIA/nsight-perf-sdk-for-ai)
  separately on the compute node. The user or administrator owns their
  installation, configuration, permissions, and lifecycle.

PyTorch does not download, install, start, or configure the service. AON builds
require only the pinned client and its matching FlatBuffers headers, not the
aggregate SDK's samples, exporters, benchmarks, or proprietary service binary.

## Build the optional adapter

From the PyTorch repository root, initialize the new dependencies alongside
the other dependencies required by your source build:

```bash
git submodule update --init third_party/nvtrace-client third_party/nvtrace-flatbuffers
python -m pip install -e . -v --no-build-isolation \
  -Ccmake.define.KINETO_ENABLE_AON=ON \
  -Ccmake.define.KINETO_ENABLE_CUPTI_PM_SAMPLING=OFF
```

Use the usual PyTorch CUDA build configuration for your machine. This feature
is off by default and cannot be combined with Kineto's CUPTI PM-sampling
feature. The resulting installation includes `torch/lib/libkineto_aon.so`
and dependency notices under `torch/share/licenses/kineto/`.

The gitlinks pin the client and FlatBuffers revisions together. NvTrace's
generated headers require FlatBuffers 23.5.26; the adapter keeps those headers
and symbols private instead of replacing PyTorch's FlatBuffers dependency.
For local source overrides, pass both paths explicitly:

```bash
python -m pip install -e . -v --no-build-isolation \
  -Ccmake.define.KINETO_ENABLE_AON=ON \
  -Ccmake.define.KINETO_ENABLE_CUPTI_PM_SAMPLING=OFF \
  -Ccmake.define.KINETO_AON_CLIENT_SOURCE_DIR=/path/to/nvtrace-client \
  -Ccmake.define.KINETO_AON_FLATBUFFERS_SOURCE_DIR=/path/to/flatbuffers-23.5.26
```

## Configure the external service

Download {download}`nvaon_service_config.yaml <examples/nvaon_service_config.yaml>`
and point the external service at that file:

```bash
export NVIDIA_NSIGHT_PERF_CONFIG_PATH="$PWD/nvaon_service_config.yaml"
```

This example enables CUDA driver tracing, GPU timing and clock calibration,
and PM collection using the SDK's `ReducedOverview.config`. Keep that metric
file beside the installed `nvidia-nsight-perf` executable. The same service
configuration works with `aon_torch_profiler.py` and the raw TSF example.
PyTorch does not load this YAML or start the service itself.

`CudaApiTrace.traceRuntimeApiCalls` is `false` in the example, so expect CUDA
driver calls and kernels. Set it to `true` if CUDA runtime API events are also
needed; runtime tracing can increase service startup time and memory usage.
PM samples are retained in raw TSF; numeric PM counter events are not yet
exported into the Kineto trace.

## Export a Kineto trace

After starting the external service and checking readiness, select the backend
before importing PyTorch:

```bash
KINETO_GPU_PROFILER=aon python aon_torch_profiler.py --output aon-trace.json
```

Download {download}`aon_torch_profiler.py <examples/aon_torch_profiler.py>`.
The example warms up CUDA, records matrix multiplications with `torch.profiler`,
and exports a Chrome trace. It fails if the trace has no timed CUDA kernels,
so a CPU-only capture cannot be mistaken for a successful GPU capture.

The converter reads the closed TSF database, selects the capturing PID and PID
namespace, calibrates CPU and GPU clocks to UTC nanoseconds, and emits CUDA
runtime calls, driver calls, kernels, copies, and sets. Kernel names are
demangled when available. Driver calls and GPU operations share correlation
IDs; CPU-operation links use the launching thread's active PyTorch correlation
stack at API submission time. Original 64-bit AON correlation and stream IDs
are preserved in activity metadata.

Only complete events within the profiler capture window are emitted. Records
without valid timing or a known device calibration are skipped. The service
can collect PM counters alongside CUDA tracing, and raw TSF captures retain
those records. This initial Kineto converter does not export numeric PM
counter events, NVTX ranges, or host callbacks, and does not support dynamic
collection toggling. CUDA graph kernels and memory nodes are decoded
when the SDK resolves their node parameters; unresolved graph nodes are skipped.

## Check service readiness and raw capture

Start the externally installed service before CUDA initializes in the workload.
For a user-local service, run as the normal job user. Root, including remapped
root inside a container, selects the SDK's global scope instead; setting
`XDG_RUNTIME_DIR` does not turn a root-run service into a user-local service.

The service, readiness tool, and workload must share a private node-local
`XDG_RUNTIME_DIR` owned by the user:

```bash
export NSIGHT_PERF_BIN_PATH=/path/to/nsight-perf/bin
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/pytorch-aon.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"
"$NSIGHT_PERF_BIN_PATH/nvidia-nsight-perf" > daemon.log 2>&1 &
daemon_pid=$!
trap 'kill -TERM "$daemon_pid" 2>/dev/null; wait "$daemon_pid" 2>/dev/null' EXIT
"$NSIGHT_PERF_BIN_PATH/nvtrace_poll_until_service_ready" --timeout-sec 60
python aon_capture.py --output ./aon-capture --captures 2 --seconds 2
```

Download {download}`aon_capture.py <examples/aon_capture.py>` before running
these commands. It loads the adapter shipped with the active PyTorch
installation, checks service connectivity, and performs two independent
streaming captures around CUDA matrix multiplications. Each capture has its
own database directory. Existing output directories are rejected.

The adapter's connectivity probe confirms that the caller can open an observer
session. It does not establish that every GPU provider is ready. Use the SDK
readiness utility and inspect `daemon.log` for initialization failures.

Inspect the resulting databases with the matching SDK tools:

```bash
"$NSIGHT_PERF_BIN_PATH/nvtrace_perfetto_trace_exporter" \
  --tsf-database-dir ./aon-capture/capture-0
```

Check the exported trace for the expected workload, CUDA API calls, and GPU
timings. Provider readiness, a successful client connection, and allocated
database bytes are separate checks; none alone guarantees a complete trace.

## Slurm example

Download {download}`aon_slurm.sh <examples/aon_slurm.sh>` beside
`aon_torch_profiler.py` and `nvaon_service_config.yaml`. Select the account and
partition appropriate to your cluster:

```bash
export NSIGHT_PERF_BIN_PATH=/shared/path/to/nsight-perf/bin
export AON_PYTHON=/shared/path/to/pytorch-venv/bin/python
export AON_EXAMPLE_SCRIPT="$PWD/aon_torch_profiler.py"
export NVIDIA_NSIGHT_PERF_CONFIG_PATH="$PWD/nvaon_service_config.yaml"
srun --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 \
  --mem=32G --time=00:10:00 bash aon_slurm.sh
```

The example starts one user-local service for a single-node, single-task,
single-GPU job, waits for readiness, exports `aon-JOB_ID/aon-trace.json` through
`torch.profiler`, and stops its own service on exit. It preserves service and
readiness logs under `aon-JOB_ID`. It does not modify the PyTorch installation
or system services. For a multi-task job, use one service per node and share the runtime
directory with that node's consumers. GPU PM-counter collection is exclusive
per GPU; coordinate with the cluster administrator when a service already owns
the allocated GPU.

To collect raw TSF databases instead, download `aon_capture.py`, set these
variables, and run the same `srun` command:

```bash
export AON_CAPTURE_MODE=raw
export AON_EXAMPLE_SCRIPT="$PWD/aon_capture.py"
```

The service's PM configuration is independent of the selected example. PM
collection can remain enabled alongside CUDA tracing. For a custom service
configuration, export `NVIDIA_NSIGHT_PERF_CONFIG_PATH` before running `srun`. Use
a capture lasting several sampling/readback intervals when checking PM data;
a very short CUDA workload can finish before PM samples become available.

## Driver and capture troubleshooting

- **Adapter missing:** use the same Python environment in which you built the
  AON-enabled PyTorch installation. A regular wheel may not contain the adapter.
- **Service unavailable:** check that the service and consumer run on the same
  node, resolve the same runtime directory, and have the required permissions.
  AON selection emits a warning when the service cannot be reached.
- **Service connected, but no CUDA API or kernel records:** check the driver
  versions and enabled providers. GPU PM samples can be available even when
  the workload's CUDA API provider is unavailable. A TSF file is not evidence
  that its expected record categories exist.
- **Containers:** run the user-local service as the normal job user and expose
  the allocated GPU devices. Make the host's `libnvidia-api.so.1` available
  inside the container; on x86 Linux this may require a read-only bind mount of
  `/usr/lib/x86_64-linux-gnu/libnvidia-api.so.1` at the same path. Verify the
  workload loads the intended `libcuda.so` and NVAPI libraries. A functioning
  CUDA workload alone does not verify that the driver's AON provider can attach.
- **CUDA compatibility libraries:** a newer workload user-mode driver may be
  needed on an older host driver. GPU GR-context tracing has a separate
  host-driver matching requirement. The service and workload can use different
  `libcuda.so` libraries; validate the actual loaded libraries and the resulting
  correlation. See the SDK's
  [driver-matching guidance](https://github.com/NVIDIA/nsight-perf-daemon#gpu-gr-context-tracing-and-cuda-forward-compatibility).
  The Slurm example accepts `AON_DAEMON_DRIVER_DIR` and
  `AON_WORKLOAD_DRIVER_DIR` for these separate search paths. It does not
  download or install compatibility libraries.
- **CPU-only `torch.profiler` output in AON mode:** check the raw TSF for CUDA
  API and GPU timing records, not just PM samples. Missing calibration,
  unresolved process identity, and out-of-window records also prevent activity
  emission. A service connection alone is insufficient. Unset
  `KINETO_GPU_PROFILER` to return to CUPTI profiling.
