/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "IActivityProfiler.h"
#include "aon/AonClientAdapter.h"

namespace KINETO_NAMESPACE {

class IAonTraceConsumer {
 public:
  virtual ~IAonTraceConsumer() = default;
  [[nodiscard]] virtual bool initialize(
      const std::filesystem::path& databasePath) = 0;
  [[nodiscard]] virtual bool start() = 0;
  [[nodiscard]] virtual bool stop() = 0;
  [[nodiscard]] virtual bool read(
      KinetoAonEventCallback callback,
      void* context,
      std::string& error) = 0;
};

class AonActivitySession final : public libkineto::IActivityProfilerSession {
 public:
  AonActivitySession();
  AonActivitySession(
      std::unique_ptr<IAonTraceConsumer> consumer,
      std::filesystem::path databasePath,
      bool cleanupDatabase = true,
      std::set<libkineto::ActivityType> activities = {});
  ~AonActivitySession() override;

  [[nodiscard]] bool prepare();
  void start() override;
  void stop() override;
  [[nodiscard]] std::vector<std::string> errors() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  void processTrace(
      libkineto::ActivityLogger& logger,
      libkineto::getLinkedActivityCallback getLinkedActivity,
      int64_t startTime,
      int64_t endTime) override;
  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;
  [[nodiscard]] std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  [[nodiscard]] std::vector<libkineto::ResourceInfo> getResourceInfos()
      override;
  [[nodiscard]] std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer()
      override;

 private:
  void recordError(std::string error);
  struct CorrelationPoint {
    int64_t timestamp;
    uint64_t id;
  };
  struct ThreadCorrelations {
    std::vector<uint64_t> stack;
    std::vector<CorrelationPoint> history;
  };

  std::unique_ptr<IAonTraceConsumer> consumer_;
  std::filesystem::path databasePath_;
  bool cleanupDatabase_{true};
  bool prepared_{false};
  bool started_{false};
  bool stopped_{false};
  bool processed_{false};
  std::set<libkineto::ActivityType> activities_;
  std::unique_ptr<libkineto::CpuTraceBuffer> traceBuffer_;
  std::vector<libkineto::ResourceInfo> resources_;
  std::mutex correlationMutex_;
  std::map<uint32_t, ThreadCorrelations> correlations_;
  std::vector<std::string> errors_;
};

// Child profiler for the externally managed NVIDIA Nsight Perf service.
class AonActivityProfiler final : public libkineto::IActivityProfiler {
 public:
  [[nodiscard]] const std::string& name() const override;
  [[nodiscard]] const std::set<libkineto::ActivityType>& availableActivities()
      const override;
  [[nodiscard]] std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
  [[nodiscard]] std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t startTimeMs,
      int64_t durationMs,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
};

// The default is CUPTI. AON must be selected explicitly with
// KINETO_GPU_PROFILER=aon.
[[nodiscard]] bool isAonGpuProfilerSelected();

// A successful connection does not guarantee that every GPU provider is ready.
[[nodiscard]] bool isAonServiceAvailable();

} // namespace KINETO_NAMESPACE
