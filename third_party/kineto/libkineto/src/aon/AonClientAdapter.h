/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#define KINETO_AON_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

// Keep SDK headers, C++ objects, and FlatBuffers inside the adapter library.
typedef struct KinetoAonConsumer KinetoAonConsumer;

typedef enum KinetoAonEventKind {
  KINETO_AON_RUNTIME,
  KINETO_AON_DRIVER,
  KINETO_AON_KERNEL,
  KINETO_AON_MEMCPY,
  KINETO_AON_MEMSET
} KinetoAonEventKind;

typedef struct KinetoAonEvent {
  KinetoAonEventKind kind;
  uint64_t start_ns;
  uint64_t end_ns;
  uint64_t launch_ns;
  uint64_t correlation_id;
  uint64_t api_correlation_id;
  uint64_t stream_id;
  uint64_t bytes;
  uint32_t process_id;
  uint32_t thread_id;
  uint32_t device_id;
  uint32_t is_graph;
  uint32_t grid[3];
  uint32_t block[3];
  // Borrowed until the callback returns; callers must copy the name.
  const char* name;
} KinetoAonEvent;

// Return zero to abort reading. No exception may escape the callback.
typedef int (
    *KinetoAonEventCallback)(void* context, const KinetoAonEvent* event);

// Read a closed TSF database. PID and namespace identify the capturing process;
// zero PID explicitly selects all processes for offline diagnostics.
KINETO_AON_API int kineto_aon_trace_read(
    const char* database_path,
    uint32_t process_id,
    uint64_t namespace_device,
    uint64_t namespace_inode,
    KinetoAonEventCallback callback,
    void* context,
    char* error,
    size_t error_size);

// Checks whether the caller can connect; does not start or configure a daemon.
KINETO_AON_API int kineto_aon_service_available(void);
KINETO_AON_API KinetoAonConsumer* kineto_aon_consumer_create(void);
KINETO_AON_API int kineto_aon_consumer_initialize(
    KinetoAonConsumer* consumer,
    const char* database_path);
KINETO_AON_API int kineto_aon_consumer_start(KinetoAonConsumer* consumer);
KINETO_AON_API int kineto_aon_consumer_stop(KinetoAonConsumer* consumer);
KINETO_AON_API void kineto_aon_consumer_destroy(KinetoAonConsumer* consumer);

#ifdef __cplusplus
}
#endif

#undef KINETO_AON_API
