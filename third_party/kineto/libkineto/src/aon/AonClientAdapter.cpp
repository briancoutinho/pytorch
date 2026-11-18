/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "AonClientAdapter.h"

#include <filesystem>

#include <NvTrace/Target/Consumer/NvTraceStreamDatabaseWriter.h>
#include <NvTrace/Target/Consumer/NvTraceStreamingConsumer.h>
#include <NvTrace/Target/Router/NvTraceRouterObserverClient.h>

struct KinetoAonConsumer {
  using Consumer = nv::trace::StreamingConsumer<
      nv::trace::TsfDiskDatabaseBufferedStorageWriter>;
  Consumer consumer;
};

extern "C" {

int kineto_aon_service_available(void) {
  try {
    nv::trace::RouterObserverClient client;
    NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint calibration{};
    return client.Initialize(nullptr, nullptr, nullptr, 1000, &calibration);
  } catch (...) {
    return 0;
  }
}

KinetoAonConsumer* kineto_aon_consumer_create(void) {
  try {
    return new KinetoAonConsumer;
  } catch (...) {
    return nullptr;
  }
}

int kineto_aon_consumer_initialize(
    KinetoAonConsumer* consumer,
    const char* database_path) {
  if (!consumer || !database_path || !*database_path) {
    return 0;
  }
  try {
    KinetoAonConsumer::Consumer::Options options;
    return consumer->consumer.Initialize(
        options,
        [database_path](nv::trace::TsfDiskDatabaseBufferedStorageWriter& writer) {
          return writer.Initialize(std::filesystem::path{database_path});
        });
  } catch (...) {
    return 0;
  }
}

int kineto_aon_consumer_start(KinetoAonConsumer* consumer) {
  try {
    return consumer && consumer->consumer.Start();
  } catch (...) {
    return 0;
  }
}

int kineto_aon_consumer_stop(KinetoAonConsumer* consumer) {
  try {
    return consumer && consumer->consumer.Stop();
  } catch (...) {
    return 0;
  }
}

void kineto_aon_consumer_destroy(KinetoAonConsumer* consumer) {
  delete consumer;
}

} // extern "C"
