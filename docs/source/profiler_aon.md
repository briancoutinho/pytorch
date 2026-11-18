# Experimental NVIDIA AON client

NVIDIA Nsight Perf SDK for AI provides always-on (AON) GPU tracing through an
external service. This experimental Kineto integration compiles its public
NvTrace client into a private shared adapter, `libkineto_aon.so`.

```{warning}
This integration is a development scaffold. The AON-to-Kineto activity
converter is not implemented. Setting `KINETO_GPU_PROFILER=aon` disables
CUPTI GPU collection, but does **not** add AON GPU events to
`torch.profiler` traces. Keep the default CUPTI backend for normal
`torch.profiler` GPU traces.
```

The examples below test the private adapter directly and write an SDK Trace
Stream Database (TSF). This diagnostic interface is not a supported public
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

## Check service readiness and capture

Start the externally installed service before the workload. For a user-local
service, the service, readiness tool, and workload must share a private
node-local `XDG_RUNTIME_DIR` owned by the user:

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

Download {download}`aon.slurm <examples/aon.slurm>` beside `aon_capture.py`.
Select the account and partition appropriate to your cluster:

```bash
export NSIGHT_PERF_BIN_PATH=/shared/path/to/nsight-perf/bin
export AON_PYTHON=/shared/path/to/pytorch-venv/bin/python
export AON_EXAMPLE_SCRIPT="$PWD/aon_capture.py"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION aon.slurm
```

The example starts one user-local service for a single-node, single-task,
single-GPU job, waits for readiness, captures through the installed adapter,
and stops its own service on exit. It preserves service and readiness logs
under `aon-JOB_ID`. It does not modify the PyTorch installation or system
services. For a multi-task job, use one service per node and share the runtime
directory with that node's consumers. GPU PM-counter collection is exclusive
per GPU; coordinate with the cluster administrator when a service already owns
the allocated GPU.

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
- **CUDA compatibility libraries:** a newer workload user-mode driver may be
  needed on an older host driver. GPU GR-context tracing has a separate
  host-driver matching requirement. The service and workload can use different
  `libcuda.so` libraries; validate the actual loaded libraries and the resulting
  correlation. See the SDK's
  [driver-matching guidance](https://github.com/NVIDIA/nsight-perf-daemon#gpu-gr-context-tracing-and-cuda-forward-compatibility).
  The Slurm example accepts `AON_DAEMON_DRIVER_DIR` and
  `AON_WORKLOAD_DRIVER_DIR` for these separate search paths. It does not
  download or install compatibility libraries.
- **CPU-only `torch.profiler` output in AON mode:** this is the current scaffold
  behavior. The TSF-to-Kineto converter and PyTorch operator correlation remain
  future work. Unset `KINETO_GPU_PROFILER` to return to normal CUPTI profiling.
