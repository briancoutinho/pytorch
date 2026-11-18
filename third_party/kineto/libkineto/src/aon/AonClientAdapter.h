/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define KINETO_AON_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

// Keep SDK headers, C++ objects, and FlatBuffers inside the adapter library.
typedef struct KinetoAonConsumer KinetoAonConsumer;

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
