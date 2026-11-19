# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

foreach(header
    "${KINETO_AON_CLIENT_SOURCE_DIR}/include/NvTrace/Target/Consumer/NvTraceStreamingConsumer.h"
    "${KINETO_AON_FLATBUFFERS_SOURCE_DIR}/include/flatbuffers/flatbuffers.h")
  if(NOT EXISTS "${header}")
    message(FATAL_ERROR
      "AON dependency is missing: ${header}. Initialize the nvtrace-client and "
      "nvtrace-flatbuffers submodules, or set KINETO_AON_CLIENT_SOURCE_DIR and "
      "KINETO_AON_FLATBUFFERS_SOURCE_DIR to local source trees.")
  endif()
endforeach()

find_package(Threads REQUIRED)
add_library(kineto_aon SHARED AonClientAdapter.cpp AonTraceReader.cpp)
set_target_properties(kineto_aon PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  INTERPROCEDURAL_OPTIMIZATION OFF)
# A separate DSO and a local-symbol map isolate FlatBuffers from PyTorch's copy.
target_include_directories(kineto_aon SYSTEM PRIVATE
  "${KINETO_AON_CLIENT_SOURCE_DIR}/include"
  "${KINETO_AON_FLATBUFFERS_SOURCE_DIR}/include")
target_link_libraries(kineto_aon PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(kineto_aon PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_property(TARGET kineto_aon APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")

install(TARGETS kineto_aon EXPORT kinetoLibraryConfig
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
  "${KINETO_AON_CLIENT_SOURCE_DIR}/LICENSE"
  "${KINETO_AON_CLIENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
  "${KINETO_AON_CLIENT_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/kineto/nvtrace-client)
install(FILES "${KINETO_AON_FLATBUFFERS_SOURCE_DIR}/LICENSE"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/kineto/nvtrace-flatbuffers)
