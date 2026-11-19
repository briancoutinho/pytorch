#!/bin/bash
set -euo pipefail

: "${NSIGHT_PERF_BIN_PATH:?Set the directory containing the SDK service and tools}"
: "${AON_PYTHON:?Set the Python executable for the AON-enabled PyTorch build}"
: "${AON_EXAMPLE_SCRIPT:?Set the absolute path to the selected AON example}"
capture_mode="${AON_CAPTURE_MODE:-profiler}"
case "${capture_mode}" in
  profiler|raw) ;;
  *) echo "AON_CAPTURE_MODE must be profiler or raw" >&2; exit 2 ;;
esac
output="${AON_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/aon-${SLURM_JOB_ID}"
mkdir -p "${output}"
export XDG_RUNTIME_DIR
XDG_RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-aon-${SLURM_JOB_ID}.XXXXXX")"
chmod 700 "${XDG_RUNTIME_DIR}"
service_ld="${AON_DAEMON_DRIVER_DIR:+${AON_DAEMON_DRIVER_DIR}:}${NSIGHT_PERF_BIN_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
workload_ld="${AON_WORKLOAD_DRIVER_DIR:+${AON_WORKLOAD_DRIVER_DIR}:}${LD_LIBRARY_PATH:-}"

env LD_LIBRARY_PATH="${service_ld}" \
  "${NSIGHT_PERF_BIN_PATH}/nvidia-nsight-perf" > "${output}/daemon.log" 2>&1 &
daemon_pid=$!
cleanup() {
  status=$?
  trap - EXIT
  if kill -0 "${daemon_pid}" 2>/dev/null; then
    kill -TERM "${daemon_pid}"
    wait "${daemon_pid}" || true
  fi
  exit "${status}"
}
trap cleanup EXIT

env LD_LIBRARY_PATH="${service_ld}" \
  "${NSIGHT_PERF_BIN_PATH}/nvtrace_poll_until_service_ready" --timeout-sec 60 \
  > "${output}/readiness.log" 2>&1
kill -0 "${daemon_pid}"
if [[ "${capture_mode}" == raw ]]; then
  env LD_LIBRARY_PATH="${workload_ld}" "${AON_PYTHON}" "${AON_EXAMPLE_SCRIPT}" \
    --output "${output}/captures" --captures 2 --seconds 2
else
  env LD_LIBRARY_PATH="${workload_ld}" KINETO_GPU_PROFILER=aon \
    "${AON_PYTHON}" "${AON_EXAMPLE_SCRIPT}" --output "${output}/aon-trace.json"
fi
