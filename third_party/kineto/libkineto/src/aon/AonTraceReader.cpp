/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "AonTraceDecoder.h"

#include <NvTrace/Offline/DataModel/NvTraceStreamDataModelBuilder.h>
#include <NvTrace/Offline/DataModel/NvTraceStreamDatabaseReader.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>

extern "C" int kineto_aon_trace_read(
    const char* database_path,
    uint32_t process_id,
    uint64_t namespace_device,
    uint64_t namespace_inode,
    KinetoAonEventCallback callback,
    void* context,
    char* error,
    size_t error_size) {
  if (error && error_size) {
    error[0] = '\0';
  }
  try {
    if (!database_path || !*database_path || !callback) {
      // @allow-raw-throw: Caught inside the standalone adapter C ABI.
      throw std::invalid_argument(
          "AON requires a database path and event callback");
    }
    using Reader = nv::trace::TraceStreamDatabaseReader<
        nv::trace::TsfDiskDatabaseBufferedStorageReader>;
    Reader reader;
    if (!reader.Initialize([&](auto& storage) {
          return storage.Initialize(std::filesystem::path{database_path});
        })) {
      // @allow-raw-throw: Caught inside the standalone adapter C ABI.
      throw std::runtime_error("AON could not open the TSF database");
    }
    nv::trace::Codec::DataModel<Reader> model;
    nv::trace::Codec::DataModelBuilder<Reader> builder(model);
    nv::trace::Codec::DataModelBuilder<Reader>::Options options;
    // Without finite observation bounds, the SDK discards active processes.
    if (!reader.GetDataBlockRangeTimeBoundsUtcNsec(
            reader.GetDataBlockRange(),
            options.observationStartUtcNsec,
            options.observationEndUtcNsec) ||
        !builder.Build(reader, options)) {
      // @allow-raw-throw: Caught inside the standalone adapter C ABI.
      throw std::runtime_error("AON could not decode the TSF data model");
    }
    kineto_aon::TraceDecoder<Reader>(model).decode(
        process_id, namespace_device, namespace_inode, callback, context);
    return 1;
  } catch (const std::exception& exception) {
    if (error && error_size) {
      std::snprintf(
          error, error_size, "AON TSF trace read failed: %s", exception.what());
    }
  } catch (...) {
    if (error && error_size) {
      std::snprintf(error, error_size, "AON trace conversion failed");
    }
  }
  return 0;
}
