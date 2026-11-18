/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "AonActivityProfiler.h"

#include <atomic>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

#include "Logger.h"
#include "aon/AonClientAdapter.h"
#include "libkineto.h"

namespace KINETO_NAMESPACE {
namespace {

const std::string kProfilerName{"NVIDIA Nsight Perf (AON)"};
const std::set<libkineto::ActivityType> kSupportedActivities{};

class NvTraceStreamingConsumer final : public IAonTraceConsumer {
 public:
  NvTraceStreamingConsumer() : consumer_(kineto_aon_consumer_create()) {}

  ~NvTraceStreamingConsumer() override {
    kineto_aon_consumer_destroy(consumer_);
  }

  bool initialize(const std::filesystem::path& databasePath) override {
    return kineto_aon_consumer_initialize(consumer_, databasePath.c_str());
  }

  bool start() override {
    return kineto_aon_consumer_start(consumer_);
  }

  bool stop() override {
    return kineto_aon_consumer_stop(consumer_);
  }

 private:
  KinetoAonConsumer* consumer_;
};

std::filesystem::path makeDatabasePath() {
  static std::atomic<uint64_t> sequence{0};
  return std::filesystem::temp_directory_path() / "kineto-aon" /
      (std::to_string(getpid()) + "-" +
       std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
}

} // namespace

AonActivitySession::AonActivitySession()
    : AonActivitySession(
          std::make_unique<NvTraceStreamingConsumer>(),
          makeDatabasePath()) {}

AonActivitySession::AonActivitySession(
    std::unique_ptr<IAonTraceConsumer> consumer,
    std::filesystem::path databasePath,
    bool cleanupDatabase)
    : consumer_(std::move(consumer)),
      databasePath_(std::move(databasePath)),
      cleanupDatabase_(cleanupDatabase) {}

AonActivitySession::~AonActivitySession() {
  if (started_) {
    stop();
  }
  consumer_.reset();
  if (!cleanupDatabase_ || databasePath_.empty()) {
    return;
  }
  std::error_code error;
  std::filesystem::remove_all(databasePath_, error);
  if (error) {
    LOG(WARNING) << "AON could not remove temporary TSF database '"
                 << databasePath_.string() << "': " << error.message();
  }
}

bool AonActivitySession::prepare() {
  if (prepared_) {
    return true;
  }
  try {
    if (!consumer_ || !consumer_->initialize(databasePath_)) {
      recordError(
          "AON could not connect to the external nvidia-nsight-perf service "
          "or initialize trace storage. Ask the administrator or user to "
          "start the service and check access permissions and storage.");
      return false;
    }
  } catch (const std::exception& error) {
    recordError(
        std::string{"AON failed while connecting to the external service: "} +
        error.what());
    return false;
  }
  prepared_ = true;
  status_ = libkineto::TraceStatus::READY;
  return true;
}

void AonActivitySession::start() {
  if (!prepared_) {
    recordError("AON session started before successful preparation");
    return;
  }
  if (started_) {
    return;
  }
  try {
    if (!consumer_->start()) {
      recordError("AON failed to start NvTrace streaming consumption");
      return;
    }
  } catch (const std::exception& error) {
    recordError(
        std::string{"AON failed to start NvTrace streaming consumption: "} +
        error.what());
    return;
  }
  started_ = true;
  status_ = libkineto::TraceStatus::RECORDING;
}

void AonActivitySession::stop() {
  if (!started_) {
    return;
  }
  try {
    if (!consumer_->stop()) {
      recordError("AON failed to stop NvTrace streaming consumption");
      started_ = false;
      return;
    }
  } catch (const std::exception& error) {
    recordError(
        std::string{"AON failed to stop NvTrace streaming consumption: "} +
        error.what());
    started_ = false;
    return;
  }
  started_ = false;
  status_ = libkineto::TraceStatus::PROCESSING;
}

std::vector<std::string> AonActivitySession::errors() {
  return errors_;
}

void AonActivitySession::processTrace(
    libkineto::ActivityLogger& /*logger*/) {}

std::unique_ptr<libkineto::DeviceInfo> AonActivitySession::getDeviceInfo() {
  return nullptr;
}

std::vector<libkineto::ResourceInfo> AonActivitySession::getResourceInfos() {
  return {};
}

std::unique_ptr<libkineto::CpuTraceBuffer> AonActivitySession::getTraceBuffer() {
  return nullptr;
}

void AonActivitySession::recordError(std::string error) {
  LOG(WARNING) << error;
  errors_.push_back(std::move(error));
  status_ = libkineto::TraceStatus::ERROR;
}

const std::string& AonActivityProfiler::name() const {
  return kProfilerName;
}

const std::set<libkineto::ActivityType>& AonActivityProfiler::
    availableActivities() const {
  return kSupportedActivities;
}

std::unique_ptr<libkineto::IActivityProfilerSession> AonActivityProfiler::
    configure(
        const std::set<libkineto::ActivityType>& /*activityTypes*/,
        const libkineto::Config& /*config*/) {
  return nullptr;
}

std::unique_ptr<libkineto::IActivityProfilerSession> AonActivityProfiler::
    configure(
        int64_t /*startTimeMs*/,
        int64_t /*durationMs*/,
        const std::set<libkineto::ActivityType>& activityTypes,
        const libkineto::Config& config) {
  return configure(activityTypes, config);
}

bool isAonGpuProfilerSelected() {
  const char* mode = std::getenv("KINETO_GPU_PROFILER");
  if (mode == nullptr || *mode == '\0' || std::string(mode) == "cupti") {
    return false;
  }
  if (std::string(mode) == "aon") {
    return true;
  }
  LOG(WARNING) << "Ignoring invalid KINETO_GPU_PROFILER='" << mode
               << "'; expected 'cupti' or 'aon'";
  return false;
}

bool isAonServiceAvailable() {
  return kineto_aon_service_available();
}

} // namespace KINETO_NAMESPACE
