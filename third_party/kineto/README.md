# Kineto

Kineto is a library used in the PyTorch Profiler.

The Kineto project enables:
- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

The central component of Kineto is Libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

## Libkineto

Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## Releases and Contributing
Kineto lives in the PyTorch repository and ships as part of each PyTorch release; there is no separate Kineto release.

Contributions are made as regular PyTorch pull requests, following PyTorch's [contributing guide](../../CONTRIBUTING.md). Changes under `third_party/kineto` are reviewed by the PyTorch Profiler maintainers. If you plan to contribute a new feature, please open a PyTorch issue to discuss it first.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.

## Experimental AON client

On Linux CUDA builds, `KINETO_ENABLE_AON=ON` builds the private
`libkineto_aon.so` adapter. Initialize PyTorch's `third_party/nvtrace-client`
and `third_party/nvtrace-flatbuffers` submodules before configuring. Their
gitlinks pin the client revision and its matching FlatBuffers 23.5.26 headers.
The build does not download dependencies, configure the aggregate SDK, or
build its samples, exporters, or daemon.

For a standalone Kineto build or local development, set
`KINETO_AON_CLIENT_SOURCE_DIR` and `KINETO_AON_FLATBUFFERS_SOURCE_DIR` to the
corresponding source directories. NvTrace and FlatBuffers headers are private
to the adapter. Its C interface exposes opaque consumer handles, and its
linker export map keeps third-party symbols local to that shared library.
Install and distribute the adapter beside Kineto/PyTorch's other libraries.

The administrator or user installs, configures, and starts the external
`nvidia-nsight-perf` daemon. Kineto only attempts to connect. An unavailable
or inaccessible service produces a warning when `KINETO_GPU_PROFILER=aon`
is selected. A successful connection confirms client access, not readiness
of every GPU provider. The default CUPTI path does not probe the service.

Selecting `KINETO_GPU_PROFILER=aon` uses AON instead of CUPTI GPU collection.
The private adapter decodes the closed TSF database into calibrated CUDA API,
kernel, memcpy, and memset activities. Kineto owns the resulting activities and
links GPU work to the CPU operation active when its driver API was submitted.
The converter filters by PID and PID namespace and retains original 64-bit IDs
in metadata. PM counters, NVTX, host callbacks, and dynamic collection toggling
are not supported. Kernel capture requires a compatible running service with
CUDA API and GPU timing providers. See the
[PyTorch AON guide](../../docs/source/profiler_aon.md) and examples.
