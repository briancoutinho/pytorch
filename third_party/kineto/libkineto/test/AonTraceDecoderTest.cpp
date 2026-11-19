/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "src/aon/AonTraceDecoder.h"

#include <NvTrace/Offline/DataModel/NvTraceStreamDatabaseReader.h>

#include <cstring>
#include <vector>

namespace {
using DiskReader = nv::trace::TraceStreamDatabaseReader<
    nv::trace::TsfDiskDatabaseBufferedStorageReader>;
struct FixtureReader {
  using RecordMetadata = DiskReader::RecordMetadata;
  using RecordPayloads = DiskReader::RecordPayloads;
  NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch kernel{};
  NvTrace_CUDA_ApiTrace_MemcpyBegin memcpy{};
  NvTrace_CUDA_ApiTrace_MemsetBegin memset{};
  struct GraphRecord {
    NvTrace_CUDA_ApiTrace_GraphNodeCreated created{};
    nv::trace::CudaApiTraceGraphNodeParams params{};
  } graph;
  bool truncated{false};
  RecordMetadata GetRecordMetadata(uint64_t location) {
    return {
        location == 5       ? NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated
            : location == 4 ? NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin
            : location == 3
            ? NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin
            : NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch,
        UINT64_MAX - 2};
  }
  RecordPayloads GetRecordPayloads(uint64_t location) {
    if (location == 5) {
      return {nullptr, reinterpret_cast<const uint8_t*>(&graph), sizeof(graph)};
    }
    if (location == 4) {
      return {
          nullptr, reinterpret_cast<const uint8_t*>(&memset), sizeof(memset)};
    }
    return {
        nullptr,
        location == 3 ? reinterpret_cast<const uint8_t*>(&memcpy)
                      : reinterpret_cast<const uint8_t*>(&kernel),
        truncated           ? 1
            : location == 3 ? sizeof(memcpy)
                            : sizeof(kernel)};
  }
};

class AonTraceDecoderTest : public ::testing::Test {
 protected:
  FixtureReader reader;
  nv::trace::Codec::DataModel<FixtureReader> model;
  nv::trace::Codec::ThreadEventProviderData::ThreadData nativeThread;
  std::vector<KinetoAonEvent> events;
  std::vector<std::string> names;

  void SetUp() override {
    model.m_pTraceReader = &reader;
    nv::trace::TscRtcCalibrationPoint cpu{};
    cpu.tscTick = 100;
    cpu.utcNsec = 10000;
    cpu.tscToUtcFactor = 2;
    cpu.utcToTscFactor = 0.5;
    model.cpuClockCalibration.calibrationPoints.push_back(cpu);
    auto& partition = model.gpuPartitions[1];
    partition.gpuUuid = NVTRACE_NIL_UUID;
    nv::trace::TscGpuCalibrationPoint gpu{};
    gpu.tscTick = 100;
    gpu.gpuNsec = 500;
    gpu.tscToGpuFactor = 4;
    gpu.gpuToTscFactor = 0.25;
    partition.gpuClockCalibration.calibrationPoints.push_back(gpu);
    auto& process = model.processes[1];
    process.namespacePid = 42;
    process.pidNamespaceId = {8, 9};
    auto& cuda = process.cudaApiTrace;
    nativeThread.nativeThreadId = 77;
    auto& thread = cuda.threads[UINT64_MAX - 1];
    thread.pThreadData = &nativeThread;
    nv::trace::Codec::CudaApiTraceData::DriverApiCall api;
    api.startRecordLocation = 1;
    api.endRecordLocation = 2;
    api.startTimestampTsc = 110;
    api.endTimestampTsc = 120;
    thread.driverApiCalls.push_back(api);
    cuda.functions[4].mangledName = "_Z3foov";
    cuda.devices.emplace_back();
    auto& device = cuda.devices.back();
    device.cudaDeviceOrdinal = 5;
    nv::trace::Codec::CudaApiTraceData::ImmediateKernelLaunch kernel;
    kernel.apiCorrelationId = UINT64_MAX - 4;
    kernel.startTimestampGpuNsec = 600;
    kernel.endTimestampGpuNsec = 680;
    kernel.recordLocation = 1;
    kernel.driverApiCallRef = {UINT64_MAX - 1, 0};
    device.immediateKernelLaunches.push_back(kernel);
    reader.kernel.cufunctionId = 4;
    reader.kernel.custreamId = UINT64_MAX - 3;
    reader.kernel.gridDim_x = 8;
    reader.kernel.blockDim_x = 128;
  }

  void decode(uint32_t pid = 42, uint64_t device = 9, uint64_t inode = 8) {
    kineto_aon::TraceDecoder<FixtureReader>(model).decode(
        pid,
        device,
        inode,
        [](void* context, const KinetoAonEvent* event) {
          auto& self = *static_cast<AonTraceDecoderTest*>(context);
          self.events.push_back(*event);
          self.names.emplace_back(event->name);
          return 1;
        },
        this);
  }
};

TEST_F(AonTraceDecoderTest, CalibratesClocksAndPreservesIdentity) {
  decode();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0].kind, KINETO_AON_DRIVER);
  EXPECT_EQ(events[0].start_ns, 10020);
  EXPECT_EQ(events[0].end_ns, 10040);
  EXPECT_EQ(events[1].kind, KINETO_AON_KERNEL);
  EXPECT_EQ(events[1].start_ns, 10050);
  EXPECT_EQ(events[1].end_ns, 10090);
  EXPECT_EQ(events[1].launch_ns, 10020);
  EXPECT_EQ(events[1].thread_id, 77);
  EXPECT_EQ(events[1].device_id, 5);
  EXPECT_EQ(events[1].stream_id, UINT64_MAX - 3);
  EXPECT_EQ(events[1].api_correlation_id, UINT64_MAX - 4);
  EXPECT_EQ(events[1].correlation_id, events[0].correlation_id);
  EXPECT_EQ(events[1].grid[0], 8);
  EXPECT_EQ(events[1].block[0], 128);
  EXPECT_EQ(names[1], "foo()");
}

TEST_F(AonTraceDecoderTest, FiltersPidAndNamespace) {
  decode(43);
  decode(42, 9, 7);
  decode(42, 10, 8);
  EXPECT_TRUE(events.empty());
  decode(0);
  EXPECT_EQ(events.size(), 2);
}

TEST_F(AonTraceDecoderTest, DoesNotInventMissingOrInvertedTimings) {
  auto& kernels =
      model.processes[1].cudaApiTrace.devices[0].immediateKernelLaunches;
  kernels[0].startTimestampGpuNsec = 0;
  decode();
  ASSERT_EQ(events.size(), 1);
  events.clear();
  kernels[0].startTimestampGpuNsec = 700;
  decode();
  ASSERT_EQ(events.size(), 1);
  events.clear();
  model.cpuClockCalibration.calibrationPoints.clear();
  decode();
  EXPECT_TRUE(events.empty());
}

TEST_F(AonTraceDecoderTest, MemoryKernelsAreClassifiedAsCopies) {
  auto& kernel =
      model.processes[1].cudaApiTrace.devices[0].immediateKernelLaunches[0];
  kernel.recordLocation = 3;
  reader.memcpy.byteCount = UINT64_C(1) << 40;
  reader.memcpy.custreamId = 17;
  decode();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[1].kind, KINETO_AON_MEMCPY);
  EXPECT_EQ(events[1].bytes, UINT64_C(1) << 40);
  EXPECT_EQ(events[1].stream_id, 17);
}

TEST_F(AonTraceDecoderTest, DecodesCopyEngineMemset) {
  auto& device = model.processes[1].cudaApiTrace.devices[0];
  device.immediateKernelLaunches.clear();
  nv::trace::Codec::CudaApiTraceData::GpuCommand command;
  command.commandType =
      nv::trace::Codec::CudaApiTraceData::GpuCommandType::EagerMemset;
  command.recordLocation = 4;
  command.startTimestampGpuNsec = 600;
  command.endTimestampGpuNsec = 680;
  command.driverApiCallRef = {UINT64_MAX - 1, 0};
  device.gpuCommands.push_back(command);
  reader.memset.byteCount = 8192;
  reader.memset.custreamId = 19;
  decode();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[1].kind, KINETO_AON_MEMSET);
  EXPECT_EQ(events[1].bytes, 8192);
  EXPECT_EQ(events[1].stream_id, 19);
  EXPECT_EQ(events[1].correlation_id, events[0].correlation_id);
}

TEST_F(AonTraceDecoderTest, DecodesGraphKernelAndSkipsUnresolvedNodes) {
  auto& cuda = model.processes[1].cudaApiTrace;
  auto& device = cuda.devices[0];
  device.immediateKernelLaunches.clear();
  cuda.recordVersionInfo.graphNodeCreatedSize =
      offsetof(FixtureReader::GraphRecord, params);
  cuda.recordVersionInfo.graphNodeCreatedMinorVersion =
      NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A;
  cuda.recordVersionInfo.graphNodeKernelParamsMinorVersion =
      NvTrace_CUDA_ApiTrace_GraphNode_KernelParams_MINOR_VERSION_A;
  reader.graph.created.graphNodeType =
      NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel;
  reader.graph.params.kernel.cufunctionId = 4;
  reader.graph.params.kernel.gridDimX = 16;
  cuda.graphNodes[123].recordLocation = 5;
  nv::trace::Codec::CudaApiTraceData::GraphLaunch launch;
  launch.source = nv::trace::Codec::CudaApiTraceData::GraphLaunchSource::Host;
  launch.payload.host.driverApiCallRef = {UINT64_MAX - 1, 0};
  launch.payload.host.custreamId = 17;
  cuda.graphLaunches.push_back(launch);
  nv::trace::Codec::CudaApiTraceData::GraphNodeLaunch node;
  node.startTimestampGpuNsec = 600;
  node.endTimestampGpuNsec = 680;
  node.graphLaunchIdx = 0;
  node.graphNodeCorrelationId = 123;
  device.graphNodeLaunches.push_back(node);
  node.graphNodeCorrelationId = 456;
  device.graphNodeLaunches.push_back(node);
  decode();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[1].kind, KINETO_AON_KERNEL);
  EXPECT_EQ(events[1].is_graph, 1);
  EXPECT_EQ(events[1].grid[0], 16);
  EXPECT_EQ(events[1].correlation_id, events[0].correlation_id);
  EXPECT_EQ(names[1], "foo()");
}

TEST_F(AonTraceDecoderTest, RejectsTruncatedPayloadAndCallbackFailure) {
  reader.truncated = true;
  EXPECT_THROW(decode(), std::runtime_error);
  reader.truncated = false;
  EXPECT_THROW(
      kineto_aon::TraceDecoder<FixtureReader>(model).decode(
          0, 0, 0, [](void*, const KinetoAonEvent*) { return 0; }, nullptr),
      std::runtime_error);
}
} // namespace
