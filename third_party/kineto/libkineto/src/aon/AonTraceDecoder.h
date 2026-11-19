/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "AonClientAdapter.h"

#include <NvTrace/Offline/DataModel/NvTraceDataModel.h>

#include <cxxabi.h>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

namespace kineto_aon {

// SDK and FlatBuffers types in this file remain private to libkineto_aon.
template <typename Reader>
class TraceDecoder {
 public:
  explicit TraceDecoder(const nv::trace::Codec::DataModel<Reader>& model)
      : model_(model) {}

  void decode(
      uint32_t pid,
      uint64_t namespaceDevice,
      uint64_t namespaceInode,
      KinetoAonEventCallback callback,
      void* context) {
    for (const auto& entry : model_.processes) {
      const auto& process = entry.second;
      if (pid &&
          (process.namespacePid != static_cast<int64_t>(pid) ||
           (namespaceInode &&
            (process.pidNamespaceId.inode != namespaceInode ||
             process.pidNamespaceId.device != namespaceDevice)))) {
        continue;
      }
      const auto& cuda = process.cudaApiTrace;
      auto baseEvent = [&](uint64_t uniqueThreadId) {
        KinetoAonEvent event{};
        event.process_id = process.namespacePid;
        event.stream_id = UINT64_MAX;
        event.device_id = UINT32_MAX;
        const auto thread = cuda.threads.find(uniqueThreadId);
        if (thread != cuda.threads.end() && thread->second.pThreadData) {
          event.thread_id = thread->second.pThreadData->nativeThreadId;
        }
        return event;
      };
      auto linkDriver = [&](KinetoAonEvent& event,
                            const Cuda::DriverApiCallRef& ref) {
        const auto thread = cuda.threads.find(ref.uniqueThreadId);
        if (thread == cuda.threads.end() ||
            ref.driverApiCallIdx >= thread->second.driverApiCalls.size()) {
          return;
        }
        const auto& api = thread->second.driverApiCalls[ref.driverApiCallIdx];
        if (!validApi(api)) {
          return;
        }
        event.correlation_id =
            correlation(entry.first, ref.uniqueThreadId, ref.driverApiCallIdx);
        event.launch_ns = model_.ConvertTscTickToUtcNsec(api.startTimestampTsc);
        if (thread->second.pThreadData) {
          event.thread_id = thread->second.pThreadData->nativeThreadId;
        }
      };
      auto emit = [&](KinetoAonEvent event, const std::string& name) {
        if (!event.start_ns || event.end_ns < event.start_ns ||
            event.end_ns > static_cast<uint64_t>(INT64_MAX)) {
          return;
        }
        event.name = name.c_str();
        if (!callback(context, &event)) {
          // @allow-raw-throw: Caught inside the standalone adapter C ABI.
          throw std::runtime_error(
              "AON event callback aborted trace conversion");
        }
      };
      for (const auto& threadEntry : cuda.threads) {
        const auto& thread = threadEntry.second;
        for (const auto& api : thread.runtimeApiCalls) {
          if (!validApi(api)) {
            continue;
          }
          auto event = baseEvent(threadEntry.first);
          event.kind = KINETO_AON_RUNTIME;
          event.start_ns =
              model_.ConvertTscTickToUtcNsec(api.startTimestampTsc);
          event.end_ns = model_.ConvertTscTickToUtcNsec(api.endTimestampTsc);
          event.launch_ns = event.start_ns;
          const auto id = static_cast<unsigned>(api.runtimeApiFunctionId);
          using namespace nv::trace::FB::Provider::CudaApiTrace;
          emit(
              event,
              id <= CudaRuntimeApiFunctionId_MAX
                  ? EnumNamesCudaRuntimeApiFunctionId()[id]
                  : "unknown CUDA runtime API");
        }
        for (size_t index = 0; index < thread.driverApiCalls.size(); ++index) {
          const auto& api = thread.driverApiCalls[index];
          if (!validApi(api)) {
            continue;
          }
          auto event = baseEvent(threadEntry.first);
          event.kind = KINETO_AON_DRIVER;
          event.start_ns =
              model_.ConvertTscTickToUtcNsec(api.startTimestampTsc);
          event.end_ns = model_.ConvertTscTickToUtcNsec(api.endTimestampTsc);
          event.launch_ns = event.start_ns;
          event.correlation_id =
              correlation(entry.first, threadEntry.first, index);
          event.api_correlation_id =
              model_.GetRecordMetadata(api.startRecordLocation).correlationId;
          const auto id = static_cast<unsigned>(api.driverApiFunctionId);
          using namespace nv::trace::FB::Provider::CudaApiTrace;
          emit(
              event,
              id <= CudaDriverApiFunctionId_MAX
                  ? EnumNamesCudaDriverApiFunctionId()[id]
                  : "unknown CUDA driver API");
        }
      }
      for (const auto& device : cuda.devices) {
        const auto* partition =
            model_.FindGpuPartitionByGpuUuid(device.gpuUuid);
        if (!partition) {
          continue;
        }
        auto gpuEvent = [&](uint64_t start, uint64_t end) {
          auto event = baseEvent(0);
          event.device_id = device.cudaDeviceOrdinal;
          event.start_ns = model_.ConvertGpuNsecToUtcNsec(*partition, start);
          event.end_ns = model_.ConvertGpuNsecToUtcNsec(*partition, end);
          return event;
        };
        auto memoryEvent = [&](KinetoAonEvent& event, uint64_t location) {
          const auto type = model_.GetCudaApiTraceRecordType(location);
          if (type == NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin) {
            const auto& params =
                payload<NvTrace_CUDA_ApiTrace_MemcpyBegin>(location);
            event.kind = KINETO_AON_MEMCPY;
            event.stream_id = params.custreamId;
            event.bytes = params.byteCount;
          } else if (type == NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin) {
            const auto& params =
                payload<NvTrace_CUDA_ApiTrace_MemsetBegin>(location);
            event.kind = KINETO_AON_MEMSET;
            event.stream_id = params.custreamId;
            event.bytes = params.byteCount;
          } else if (
              type == NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyBegin ||
              type == NvTrace_CUDA_ApiTrace_RecordType_GraphMemsetBegin) {
            const bool copy =
                type == NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyBegin;
            uint64_t nodeId;
            if (copy) {
              const auto& params =
                  payload<NvTrace_CUDA_ApiTrace_GraphMemcpyBegin>(location);
              nodeId = params.graphNodeCorrelationId;
              event.stream_id = params.custreamId;
            } else {
              const auto& params =
                  payload<NvTrace_CUDA_ApiTrace_GraphMemsetBegin>(location);
              nodeId = params.graphNodeCorrelationId;
              event.stream_id = params.custreamId;
            }
            const auto* node = model_.FindCuGraphNode(cuda, nodeId);
            if (!node) {
              return false;
            }
            const auto nodeType =
                model_.GetCudaApiTraceGraphNodeType(node->recordLocation, cuda);
            if (nodeType !=
                (copy ? NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy
                      : NvTrace_CUDA_ApiTrace_GraphNodeType_Memset)) {
              return false;
            }
            const auto* params = model_.GetCudaApiTraceGraphNodeParams(
                node->recordLocation, cuda, nodeType);
            if (!params) {
              return false;
            }
            event.kind = copy ? KINETO_AON_MEMCPY : KINETO_AON_MEMSET;
            event.is_graph = 1;
            event.bytes =
                copy ? params->memcpy.byteCount : params->memset.byteCount;
          } else {
            return false;
          }
          emit(event, event.kind == KINETO_AON_MEMCPY ? "Memcpy" : "Memset");
          return true;
        };
        for (const auto& kernel : device.immediateKernelLaunches) {
          auto event = gpuEvent(
              kernel.startTimestampGpuNsec, kernel.endTimestampGpuNsec);
          linkDriver(event, kernel.driverApiCallRef);
          event.api_correlation_id = kernel.apiCorrelationId;
          const auto type =
              model_.GetCudaApiTraceRecordType(kernel.recordLocation);
          if (type != NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch) {
            memoryEvent(event, kernel.recordLocation);
            continue;
          }
          const auto& params =
              payload<NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch>(
                  kernel.recordLocation);
          event.kind = KINETO_AON_KERNEL;
          event.stream_id = params.custreamId;
          event.grid[0] = params.gridDim_x;
          event.grid[1] = params.gridDim_y;
          event.grid[2] = params.gridDim_z;
          event.block[0] = params.blockDim_x;
          event.block[1] = params.blockDim_y;
          event.block[2] = params.blockDim_z;
          emit(event, kernelName(cuda, params.cufunctionId));
        }
        for (const auto& command : device.gpuCommands) {
          if (command.commandType != Cuda::GpuCommandType::EagerMemcpy &&
              command.commandType != Cuda::GpuCommandType::EagerMemset &&
              command.commandType != Cuda::GpuCommandType::GraphMemcpy &&
              command.commandType != Cuda::GpuCommandType::GraphMemset) {
            continue;
          }
          auto event = gpuEvent(
              command.startTimestampGpuNsec, command.endTimestampGpuNsec);
          linkDriver(event, command.driverApiCallRef);
          event.api_correlation_id =
              model_.GetRecordMetadata(command.recordLocation).correlationId;
          memoryEvent(event, command.recordLocation);
        }
        for (const auto& node : device.graphNodeLaunches) {
          const auto* graphNode =
              model_.FindCuGraphNode(cuda, node.graphNodeCorrelationId);
          if (!graphNode) {
            continue;
          }
          const auto type = model_.GetCudaApiTraceGraphNodeType(
              graphNode->recordLocation, cuda);
          const auto* params = model_.GetCudaApiTraceGraphNodeParams(
              graphNode->recordLocation, cuda, type);
          if (!params) {
            continue;
          }
          auto event =
              gpuEvent(node.startTimestampGpuNsec, node.endTimestampGpuNsec);
          event.is_graph = 1;
          const auto* launch = model_.FindGraphLaunch(cuda, node);
          if (launch && launch->source == Cuda::GraphLaunchSource::Host) {
            linkDriver(event, launch->payload.host.driverApiCallRef);
            event.stream_id = launch->payload.host.custreamId;
            event.api_correlation_id = launch->payload.host.apiCorrelationId;
          }
          if (type == NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel) {
            event.kind = KINETO_AON_KERNEL;
            event.grid[0] = params->kernel.gridDimX;
            event.grid[1] = params->kernel.gridDimY;
            event.grid[2] = params->kernel.gridDimZ;
            event.block[0] = params->kernel.blockDimX;
            event.block[1] = params->kernel.blockDimY;
            event.block[2] = params->kernel.blockDimZ;
            emit(event, kernelName(cuda, params->kernel.cufunctionId));
          } else if (type == NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy) {
            event.kind = KINETO_AON_MEMCPY;
            event.bytes = params->memcpy.byteCount;
            emit(event, "Memcpy");
          } else if (type == NvTrace_CUDA_ApiTrace_GraphNodeType_Memset) {
            event.kind = KINETO_AON_MEMSET;
            event.bytes = params->memset.byteCount;
            emit(event, "Memset");
          }
        }
      }
    }
  }

 private:
  using Cuda = nv::trace::Codec::CudaApiTraceData;

  static bool validApi(const Cuda::ApiCall& api) {
    return api.startRecordLocation && api.endRecordLocation &&
        api.startTimestampTsc && api.endTimestampTsc >= api.startTimestampTsc;
  }

  template <typename Payload>
  const Payload& payload(uint64_t location) const {
    const auto record = model_.m_pTraceReader->GetRecordPayloads(location);
    if (!record.pMajorPayload || record.maxMajorPayloadSize < sizeof(Payload)) {
      // @allow-raw-throw: Caught inside the standalone adapter C ABI.
      throw std::runtime_error("AON record has an incomplete payload");
    }
    return *reinterpret_cast<const Payload*>(record.pMajorPayload);
  }

  uint64_t correlation(NVTRACE_LUID process, uint64_t thread, size_t index) {
    const auto key = std::make_tuple(process, thread, index);
    return correlations_.try_emplace(key, correlations_.size() + 1)
        .first->second;
  }

  static std::string kernelName(const Cuda& cuda, uint64_t function) {
    const char* name =
        nv::trace::Codec::DataModel<Reader>::GetCuFunctionMangledName(
            cuda, function);
    if (!name || !*name) {
      return "unknown CUDA kernel";
    }
    int status = 0;
    std::unique_ptr<char, decltype(&std::free)> demangled(
        abi::__cxa_demangle(name, nullptr, nullptr, &status), &std::free);
    return status == 0 && demangled ? demangled.get() : name;
  }

  const nv::trace::Codec::DataModel<Reader>& model_;
  std::map<std::tuple<NVTRACE_LUID, uint64_t, size_t>, uint64_t> correlations_;
};

} // namespace kineto_aon
