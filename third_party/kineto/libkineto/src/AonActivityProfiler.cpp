/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "AonActivityProfiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "Logger.h"
#include "aon/AonClientAdapter.h"
#include "libkineto.h"
#include "output_base.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {
namespace {

const std::string kProfilerName{"NVIDIA Nsight Perf (AON)"};
const std::set<libkineto::ActivityType> kSupportedActivities{
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::CUDA_DRIVER,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET};

class NvTraceStreamingConsumer final : public IAonTraceConsumer {
 public:
  NvTraceStreamingConsumer() : consumer_(kineto_aon_consumer_create()) {}

  ~NvTraceStreamingConsumer() override {
    kineto_aon_consumer_destroy(consumer_);
  }

  bool initialize(const std::filesystem::path& databasePath) override {
    databasePath_ = databasePath;
    struct stat ns{};
    if (::stat("/proc/self/ns/pid", &ns) != 0) {
      return false;
    }
    namespaceDevice_ = ns.st_dev;
    namespaceInode_ = ns.st_ino;
    return kineto_aon_consumer_initialize(consumer_, databasePath.c_str());
  }

  bool start() override {
    return kineto_aon_consumer_start(consumer_);
  }

  bool stop() override {
    return kineto_aon_consumer_stop(consumer_);
  }

  bool read(KinetoAonEventCallback callback, void* context, std::string& error)
      override {
    // Close the asynchronous writer before opening its database for reading.
    kineto_aon_consumer_destroy(consumer_);
    consumer_ = nullptr;
    char message[512]{};
    const bool success = kineto_aon_trace_read(
        databasePath_.c_str(),
        getpid(),
        namespaceDevice_,
        namespaceInode_,
        callback,
        context,
        message,
        sizeof(message));
    error = message;
    return success;
  }

 private:
  std::filesystem::path databasePath_;
  uint64_t namespaceDevice_{0};
  uint64_t namespaceInode_{0};
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
    bool cleanupDatabase,
    std::set<libkineto::ActivityType> activities)
    : consumer_(std::move(consumer)),
      databasePath_(std::move(databasePath)),
      cleanupDatabase_(cleanupDatabase),
      activities_(
          activities.empty() ? kSupportedActivities : std::move(activities)) {}

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
  if (stopped_ || processed_) {
    recordError("AON sessions cannot be restarted; configure a fresh session");
    return;
  }
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
  std::lock_guard<std::mutex> guard(correlationMutex_);
  started_ = true;
  status_ = libkineto::TraceStatus::RECORDING;
}

void AonActivitySession::stop() {
  if (!started_) {
    return;
  }
  {
    std::lock_guard<std::mutex> guard(correlationMutex_);
    started_ = false;
    stopped_ = true;
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

void AonActivitySession::pushCorrelationId(uint64_t id) {
  std::lock_guard<std::mutex> guard(correlationMutex_);
  if (!started_) {
    return;
  }
  auto& thread = correlations_[libkineto::systemThreadId()];
  thread.stack.push_back(id);
  thread.history.push_back(
      {libkineto::timeSinceEpoch(std::chrono::system_clock::now()), id});
}

void AonActivitySession::popCorrelationId() {
  std::lock_guard<std::mutex> guard(correlationMutex_);
  if (!started_) {
    return;
  }
  auto& thread = correlations_[libkineto::systemThreadId()];
  if (thread.stack.empty()) {
    return;
  }
  thread.stack.pop_back();
  thread.history.push_back(
      {libkineto::timeSinceEpoch(std::chrono::system_clock::now()),
       thread.stack.empty() ? 0 : thread.stack.back()});
}

void AonActivitySession::processTrace(libkineto::ActivityLogger& logger) {
  processTrace(logger, {}, 0, std::numeric_limits<int64_t>::max());
}

void AonActivitySession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback getLinkedActivity,
    int64_t startTime,
    int64_t endTime) {
  if (processed_) {
    return;
  }
  if (!stopped_ || status_ == libkineto::TraceStatus::ERROR) {
    recordError("AON trace processing requires a successfully stopped session");
    return;
  }
  processed_ = true;
  auto buffer = std::make_unique<libkineto::CpuTraceBuffer>();
  buffer->span = {startTime, endTime, "AON"};
  buffer->gpuOpCount = 0;
  struct Conversion {
    AonActivitySession& session;
    libkineto::CpuTraceBuffer& buffer;
    libkineto::getLinkedActivityCallback& linkedActivity;
    int64_t start;
    int64_t end;
    std::map<uint64_t, int32_t> correlations;
    std::map<std::pair<uint32_t, uint64_t>, int64_t> streams;
    std::set<uint32_t> devices;
    std::string error;
  } conversion{
      *this, *buffer, getLinkedActivity, startTime, endTime, {}, {}, {}, {}};
  const auto receive = [](void* context, const KinetoAonEvent* event) -> int {
    auto& state = *static_cast<Conversion*>(context);
    try {
      if (!event || !event->name || !event->start_ns ||
          event->end_ns < event->start_ns || event->end_ns > INT64_MAX) {
        state.error = "AON received an invalid event or timestamp";
        return 0;
      }
      libkineto::ActivityType type;
      switch (event->kind) {
        case KINETO_AON_RUNTIME:
          type = libkineto::ActivityType::CUDA_RUNTIME;
          break;
        case KINETO_AON_DRIVER:
          type = libkineto::ActivityType::CUDA_DRIVER;
          break;
        case KINETO_AON_KERNEL:
          type = libkineto::ActivityType::CONCURRENT_KERNEL;
          break;
        case KINETO_AON_MEMCPY:
          type = libkineto::ActivityType::GPU_MEMCPY;
          break;
        case KINETO_AON_MEMSET:
          type = libkineto::ActivityType::GPU_MEMSET;
          break;
        default:
          state.error = "AON received an unknown event kind";
          return 0;
      }
      if (!state.session.activities_.count(type) ||
          static_cast<int64_t>(event->start_ns) < state.start ||
          static_cast<int64_t>(event->end_ns) > state.end) {
        return 1;
      }
      const bool gpu = event->kind >= KINETO_AON_KERNEL;
      if (event->thread_id > INT32_MAX || event->process_id > INT32_MAX ||
          (gpu && event->device_id > INT32_MAX)) {
        state.error = "AON device or thread ID exceeds Kineto's range";
        return 0;
      }
      state.buffer.emplace_activity(
          state.buffer.span, type, std::string(event->name));
      auto& activity = *state.buffer.activities.back();
      activity.startTime = event->start_ns;
      activity.endTime = event->end_ns;
      activity.threadId = event->thread_id;
      activity.device = gpu ? event->device_id : event->process_id;
      activity.resource = event->thread_id;
      if (event->correlation_id) {
        if (state.correlations.size() >= INT32_MAX) {
          state.error = "AON exhausted Kineto correlation IDs";
          return 0;
        }
        activity.id =
            state.correlations
                .try_emplace(
                    event->correlation_id, state.correlations.size() + 1)
                .first->second;
        activity.flow.id = activity.id;
        activity.flow.type = libkineto::kLinkAsyncCpuGpu;
        activity.flow.start = event->kind == KINETO_AON_DRIVER;
      }
      activity.addMetadata("correlation", activity.id);
      activity.addMetadata("aon correlation", event->api_correlation_id);
      if (gpu) {
        state.devices.insert(event->device_id);
        activity.addMetadata("is graph", event->is_graph);
        const auto stream = std::make_pair(event->device_id, event->stream_id);
        activity.resource = event->stream_id == UINT64_MAX
            ? -1
            : state.streams.try_emplace(stream, state.streams.size() + 1)
                  .first->second;
        activity.addMetadata("stream", event->stream_id);
        activity.addMetadata("device", event->device_id);
        if (event->kind == KINETO_AON_KERNEL) {
          activity.addMetadata(
              "grid",
              fmt::format(
                  "[{},{},{}]",
                  event->grid[0],
                  event->grid[1],
                  event->grid[2]));
          activity.addMetadata(
              "block",
              fmt::format(
                  "[{},{},{}]",
                  event->block[0],
                  event->block[1],
                  event->block[2]));
        } else {
          activity.addMetadata("bytes", event->bytes);
        }
        ++state.buffer.gpuOpCount;
      }
      const auto thread = state.session.correlations_.find(event->thread_id);
      if (event->launch_ns && event->launch_ns <= INT64_MAX &&
          thread != state.session.correlations_.end()) {
        const auto& history = thread->second.history;
        auto point = std::upper_bound(
            history.begin(),
            history.end(),
            event->launch_ns,
            [](uint64_t time, const CorrelationPoint& value) {
              return time < static_cast<uint64_t>(value.timestamp);
            });
        if (point != history.begin()) {
          const uint64_t externalId = std::prev(point)->id;
          activity.addMetadata("External id", externalId);
          if (externalId && externalId <= INT32_MAX && state.linkedActivity) {
            activity.linked =
                state.linkedActivity(static_cast<int32_t>(externalId));
          }
        }
      }
      return 1;
    } catch (const std::exception& error) {
      state.error = error.what();
    } catch (...) {
      state.error = "AON activity conversion failed";
    }
    return 0;
  };
  std::string error;
  if (!consumer_->read(receive, &conversion, error) ||
      !conversion.error.empty()) {
    recordError(
        conversion.error.empty() ? "AON trace read failed: " + error
                                 : conversion.error);
    return;
  }
  std::set<int32_t> flowSources;
  std::set<int32_t> flowDestinations;
  for (const auto& activity : buffer->activities) {
    if (activity->flow.type) {
      (activity->flow.start ? flowSources : flowDestinations)
          .insert(activity->id);
    }
  }
  for (const auto& activity : buffer->activities) {
    if (!flowSources.count(activity->id) ||
        !flowDestinations.count(activity->id)) {
      activity->flow.type = 0;
    }
  }
  traceBuffer_ = std::move(buffer);
  const bool requestedGpu =
      activities_.count(libkineto::ActivityType::CONCURRENT_KERNEL) ||
      activities_.count(libkineto::ActivityType::GPU_MEMCPY) ||
      activities_.count(libkineto::ActivityType::GPU_MEMSET);
  if (requestedGpu && !traceBuffer_->gpuOpCount) {
    const std::string warning =
        "AON produced no timed GPU activities; check CUDA providers, driver compatibility, and clock calibration";
    LOG(WARNING) << warning;
    errors_.push_back(warning);
    status_ = libkineto::TraceStatus::WARNING;
  }
  for (const auto& stream : conversion.streams) {
    resources_.push_back(
        {stream.second,
         stream.second,
         stream.first.first,
         "stream " + std::to_string(stream.first.second)});
  }
  for (uint32_t device : conversion.devices) {
    logger.handleDeviceInfo(
        {device, device, "GPU " + std::to_string(device), "AON"}, startTime);
  }
  for (const auto& activity : traceBuffer_->activities) {
    logger.handleActivity(*activity);
  }
}

std::unique_ptr<libkineto::DeviceInfo> AonActivitySession::getDeviceInfo() {
  return nullptr;
}

std::vector<libkineto::ResourceInfo> AonActivitySession::getResourceInfos() {
  return resources_;
}

std::unique_ptr<libkineto::CpuTraceBuffer> AonActivitySession::
    getTraceBuffer() {
  return std::move(traceBuffer_);
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
        const std::set<libkineto::ActivityType>& activityTypes,
        const libkineto::Config& /*config*/) {
  std::set<libkineto::ActivityType> selected;
  std::set_intersection(
      activityTypes.begin(),
      activityTypes.end(),
      kSupportedActivities.begin(),
      kSupportedActivities.end(),
      std::inserter(selected, selected.end()));
  if (selected.empty()) {
    return nullptr;
  }
  auto session = std::make_unique<AonActivitySession>(
      std::make_unique<NvTraceStreamingConsumer>(),
      makeDatabasePath(),
      true,
      std::move(selected));
  if (!session->prepare()) {
    // @allow-raw-throw: Standalone Kineto cannot depend on c10.
    throw std::runtime_error(session->errors().front());
  }
  return session;
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
