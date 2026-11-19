/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "include/Config.h"
#include "include/libkineto.h"
#include "include/time_since_epoch.h"
#include "src/AonActivityProfiler.h"
#include "src/aon/AonClientAdapter.h"
#include "src/output_membuf.h"

using namespace KINETO_NAMESPACE;

namespace {

struct FakeConsumerState {
  int initializeCalls{0};
  int startCalls{0};
  int stopCalls{0};
  bool initializeResult{true};
  bool startResult{true};
  bool stopResult{true};
  std::filesystem::path initializedPath;
  std::vector<KinetoAonEvent> events;
  bool readResult{true};
  int readCalls{0};
};

class FakeAonTraceConsumer final : public IAonTraceConsumer {
 public:
  explicit FakeAonTraceConsumer(FakeConsumerState& state) : state_(state) {}

  bool initialize(const std::filesystem::path& databasePath) override {
    ++state_.initializeCalls;
    state_.initializedPath = databasePath;
    return state_.initializeResult;
  }

  bool start() override {
    ++state_.startCalls;
    return state_.startResult;
  }

  bool stop() override {
    ++state_.stopCalls;
    return state_.stopResult;
  }

  bool read(KinetoAonEventCallback callback, void* context, std::string& error)
      override {
    ++state_.readCalls;
    for (const auto& event : state_.events) {
      if (!callback(context, &event)) {
        return false;
      }
    }
    error = state_.readResult ? "" : "incomplete database";
    return state_.readResult;
  }

 private:
  FakeConsumerState& state_;
};

class ScopedGpuProfilerMode {
 public:
  explicit ScopedGpuProfilerMode(const char* value) {
    const char* oldValue = std::getenv("KINETO_GPU_PROFILER");
    if (oldValue != nullptr) {
      oldValue_ = oldValue;
      hadOldValue_ = true;
    }
    if (value == nullptr) {
      unsetenv("KINETO_GPU_PROFILER");
    } else {
      setenv("KINETO_GPU_PROFILER", value, 1);
    }
  }

  ~ScopedGpuProfilerMode() {
    if (hadOldValue_) {
      setenv("KINETO_GPU_PROFILER", oldValue_.c_str(), 1);
    } else {
      unsetenv("KINETO_GPU_PROFILER");
    }
  }

 private:
  std::string oldValue_;
  bool hadOldValue_{false};
};

} // namespace

TEST(AonActivityProfilerTest, AdvertisesDecodedActivities) {
  AonActivityProfiler profiler;
  libkineto::Config config;

  EXPECT_EQ(profiler.name(), "NVIDIA Nsight Perf (AON)");
  EXPECT_EQ(profiler.availableActivities().size(), 5);
  EXPECT_EQ(
      profiler.availableActivities().count(
          libkineto::ActivityType::CONCURRENT_KERNEL),
      1);
  EXPECT_EQ(profiler.configure({}, config), nullptr);
  EXPECT_EQ(profiler.configure(0, 1, {}, config), nullptr);
}

TEST(AonActivityProfilerTest, AonModeRequiresExplicitValidSelection) {
  {
    ScopedGpuProfilerMode mode(nullptr);
    EXPECT_FALSE(isAonGpuProfilerSelected());
  }
  {
    ScopedGpuProfilerMode mode("cupti");
    EXPECT_FALSE(isAonGpuProfilerSelected());
  }
  {
    ScopedGpuProfilerMode mode("aon");
    EXPECT_TRUE(isAonGpuProfilerSelected());
  }
  {
    ScopedGpuProfilerMode mode("invalid");
    EXPECT_FALSE(isAonGpuProfilerSelected());
  }
}

TEST(AonActivitySessionTest, DrivesConsumerLifecycleOnce) {
  FakeConsumerState state;
  AonActivitySession session(
      std::make_unique<FakeAonTraceConsumer>(state),
      std::filesystem::path{"unused"},
      false);

  ASSERT_TRUE(session.prepare());
  EXPECT_TRUE(session.prepare());
  EXPECT_EQ(state.initializeCalls, 1);

  session.start();
  session.start();
  EXPECT_EQ(state.startCalls, 1);
  EXPECT_EQ(session.status(), libkineto::TraceStatus::RECORDING);

  session.stop();
  session.stop();
  EXPECT_EQ(state.stopCalls, 1);
  EXPECT_EQ(session.status(), libkineto::TraceStatus::PROCESSING);
  EXPECT_TRUE(session.errors().empty());
}

TEST(AonActivitySessionTest, ReportsExternalServiceFailure) {
  FakeConsumerState state;
  state.initializeResult = false;
  AonActivitySession session(
      std::make_unique<FakeAonTraceConsumer>(state),
      std::filesystem::path{"unused"},
      false);

  EXPECT_FALSE(session.prepare());
  EXPECT_EQ(session.status(), libkineto::TraceStatus::ERROR);
  ASSERT_EQ(session.errors().size(), 1);
  EXPECT_NE(
      session.errors()[0].find("external nvidia-nsight-perf"),
      std::string::npos);

  session.start();
  session.stop();
  EXPECT_EQ(state.startCalls, 0);
  EXPECT_EQ(state.stopCalls, 0);
}

TEST(AonActivitySessionTest, CleansTemporaryDatabase) {
  FakeConsumerState state;
  const auto databasePath = std::filesystem::temp_directory_path() /
      "kineto-aon-session-cleanup-test";
  std::filesystem::remove_all(databasePath);
  std::filesystem::create_directories(databasePath);

  {
    AonActivitySession session(
        std::make_unique<FakeAonTraceConsumer>(state), databasePath);
    ASSERT_TRUE(session.prepare());
    EXPECT_TRUE(std::filesystem::exists(databasePath));
  }

  EXPECT_FALSE(std::filesystem::exists(databasePath));
}

TEST(AonActivitySessionTest, StopsActiveConsumerBeforeDestruction) {
  FakeConsumerState state;
  {
    AonActivitySession session(
        std::make_unique<FakeAonTraceConsumer>(state),
        std::filesystem::path{"unused"},
        false);
    ASSERT_TRUE(session.prepare());
    session.start();
  }
  EXPECT_EQ(state.stopCalls, 1);
}

TEST(AonClientAdapterTest, RejectsInvalidArgumentsWithoutConnecting) {
  EXPECT_FALSE(kineto_aon_consumer_initialize(nullptr, "unused"));
  EXPECT_FALSE(kineto_aon_consumer_start(nullptr));
  EXPECT_FALSE(kineto_aon_consumer_stop(nullptr));
  kineto_aon_consumer_destroy(nullptr);

  auto* consumer = kineto_aon_consumer_create();
  ASSERT_NE(consumer, nullptr);
  EXPECT_FALSE(kineto_aon_consumer_initialize(consumer, nullptr));
  EXPECT_FALSE(kineto_aon_consumer_initialize(consumer, ""));
  EXPECT_FALSE(kineto_aon_consumer_start(consumer));
  EXPECT_FALSE(kineto_aon_consumer_stop(consumer));
  kineto_aon_consumer_destroy(consumer);
}

namespace {
KinetoAonEvent event(KinetoAonEventKind kind, uint64_t start = 100) {
  KinetoAonEvent result{};
  result.kind = kind;
  result.start_ns = start;
  result.end_ns = start + 10;
  result.launch_ns = start;
  result.correlation_id = UINT64_MAX - 3;
  result.api_correlation_id = UINT64_MAX - 7;
  result.stream_id = UINT64_MAX - 1;
  result.device_id = 3;
  result.thread_id = libkineto::systemThreadId();
  result.process_id = libkineto::processId();
  result.name = "test kernel";
  return result;
}
} // namespace

TEST(AonActivitySessionTest, OwnsEventsAndMapsWideIdsWithoutTruncation) {
  FakeConsumerState state;
  state.events = {
      event(KINETO_AON_DRIVER),
      event(KINETO_AON_KERNEL, 120),
      event(KINETO_AON_MEMCPY, 140),
      event(KINETO_AON_MEMSET, 160)};
  state.events[2].bytes = UINT64_C(1) << 40;
  libkineto::Config config;
  libkineto::MemoryTraceLogger logger(config);
  std::unique_ptr<libkineto::CpuTraceBuffer> buffer;
  {
    AonActivitySession session(
        std::make_unique<FakeAonTraceConsumer>(state), "unused", false);
    ASSERT_TRUE(session.prepare());
    session.start();
    session.stop();
    session.processTrace(logger);
    session.processTrace(logger);
    EXPECT_EQ(state.readCalls, 1);
    ASSERT_TRUE(session.errors().empty());
    ASSERT_EQ(session.getResourceInfos().size(), 1);
    EXPECT_EQ(session.getResourceInfos()[0].deviceId, 3);
    buffer = session.getTraceBuffer();
  }
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->activities.size(), 4);
  EXPECT_EQ(buffer->gpuOpCount, 3);
  const auto& driver = *buffer->activities[0];
  const auto& kernel = *buffer->activities[1];
  EXPECT_EQ(kernel.name(), "test kernel");
  EXPECT_EQ(kernel.deviceId(), 3);
  EXPECT_EQ(kernel.resourceId(), 1);
  EXPECT_EQ(kernel.correlationId(), driver.correlationId());
  EXPECT_GT(kernel.correlationId(), 0);
  EXPECT_TRUE(driver.flowStart());
  EXPECT_FALSE(kernel.flowStart());
  EXPECT_EQ(kernel.traceSpan(), &buffer->span);
  EXPECT_EQ(
      kernel.getMetadataValue("aon correlation"),
      std::to_string(UINT64_MAX - 7));
  EXPECT_EQ(kernel.getMetadataValue("stream"), std::to_string(UINT64_MAX - 1));
  EXPECT_EQ(
      buffer->activities[2]->getMetadataValue("bytes"),
      std::to_string(UINT64_C(1) << 40));
}

TEST(AonActivitySessionTest, FiltersRequestedActivitiesAndCaptureWindow) {
  FakeConsumerState state;
  state.events = {
      event(KINETO_AON_RUNTIME, 110),
      event(KINETO_AON_KERNEL, 120),
      event(KINETO_AON_KERNEL, 1),
      event(KINETO_AON_KERNEL, 200)};
  AonActivitySession session(
      std::make_unique<FakeAonTraceConsumer>(state),
      "unused",
      false,
      {libkineto::ActivityType::CONCURRENT_KERNEL});
  libkineto::Config config;
  libkineto::MemoryTraceLogger logger(config);
  ASSERT_TRUE(session.prepare());
  session.start();
  session.stop();
  session.processTrace(logger, {}, 100, 200);
  auto buffer = session.getTraceBuffer();
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->activities.size(), 1);
  EXPECT_EQ(buffer->activities[0]->timestamp(), 120);
  EXPECT_EQ(buffer->activities[0]->flowType(), 0);
}

TEST(AonActivitySessionTest, ReadFailureDoesNotPublishPartialEvents) {
  FakeConsumerState state;
  state.events = {event(KINETO_AON_KERNEL)};
  state.readResult = false;
  AonActivitySession session(
      std::make_unique<FakeAonTraceConsumer>(state), "unused", false);
  libkineto::Config config;
  libkineto::MemoryTraceLogger logger(config);
  ASSERT_TRUE(session.prepare());
  session.start();
  session.stop();
  session.processTrace(logger);
  EXPECT_EQ(session.getTraceBuffer(), nullptr);
  EXPECT_TRUE(logger.traceActivities()->empty());
  EXPECT_EQ(session.status(), libkineto::TraceStatus::ERROR);
  ASSERT_FALSE(session.errors().empty());
  EXPECT_NE(session.errors()[0].find("incomplete database"), std::string::npos);
}

TEST(AonActivitySessionTest, RejectsInvalidTimestampsAndIds) {
  for (int invalid = 0; invalid < 3; ++invalid) {
    FakeConsumerState state;
    auto value = event(KINETO_AON_KERNEL);
    if (invalid == 0) {
      value.end_ns = value.start_ns - 1;
    } else if (invalid == 1) {
      value.end_ns = UINT64_MAX;
    } else {
      value.device_id = UINT32_MAX;
    }
    state.events.push_back(value);
    AonActivitySession session(
        std::make_unique<FakeAonTraceConsumer>(state), "unused", false);
    libkineto::Config config;
    libkineto::MemoryTraceLogger logger(config);
    ASSERT_TRUE(session.prepare());
    session.start();
    session.stop();
    session.processTrace(logger);
    EXPECT_EQ(session.getTraceBuffer(), nullptr);
    EXPECT_EQ(session.status(), libkineto::TraceStatus::ERROR);
  }
}

TEST(AonActivitySessionTest, CorrelatesDelayedGpuWorkWithNestedCpuOperations) {
  FakeConsumerState state;
  AonActivitySession session(
      std::make_unique<FakeAonTraceConsumer>(state), "unused", false);
  libkineto::Config config;
  libkineto::MemoryTraceLogger logger(config);
  libkineto::GenericTraceActivity outer;
  libkineto::GenericTraceActivity inner;
  ASSERT_TRUE(session.prepare());
  session.start();
  session.pushCorrelationId(7);
  session.pushCorrelationId(9);
  auto first = event(
      KINETO_AON_KERNEL,
      libkineto::timeSinceEpoch(std::chrono::system_clock::now()));
  session.popCorrelationId();
  auto second = event(
      KINETO_AON_KERNEL,
      libkineto::timeSinceEpoch(std::chrono::system_clock::now()));
  session.popCorrelationId();
  // Execution occurs after both CPU operations have returned.
  first.start_ns += 1000000;
  first.end_ns += 1000000;
  second.start_ns += 1000000;
  second.end_ns += 1000000;
  state.events = {first, second};
  session.stop();
  session.processTrace(
      logger,
      [&](int32_t id) -> const libkineto::ITraceActivity* {
        return id == 9 ? &inner : id == 7 ? &outer : nullptr;
      },
      0,
      INT64_MAX);
  auto buffer = session.getTraceBuffer();
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->activities.size(), 2);
  EXPECT_EQ(buffer->activities[0]->linkedActivity(), &inner);
  EXPECT_EQ(buffer->activities[1]->linkedActivity(), &outer);
}

TEST(AonClientAdapterTest, ReadErrorsStayWithinCAbi) {
  char error[128]{};
  EXPECT_FALSE(kineto_aon_trace_read(
      nullptr, 0, 0, 0, nullptr, nullptr, error, sizeof(error)));
  EXPECT_NE(std::string(error).find("database path"), std::string::npos);
  EXPECT_FALSE(kineto_aon_trace_read(
      "/nonexistent/aon-test",
      0,
      0,
      0,
      [](void*, const KinetoAonEvent*) { return 1; },
      nullptr,
      error,
      sizeof(error)));
  EXPECT_NE(std::string(error).find("TSF"), std::string::npos);
}
