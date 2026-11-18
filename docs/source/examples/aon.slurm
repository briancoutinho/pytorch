#!/bin/bash
#SBATCH --job-name=pytorch-aon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=aon-%j.log
set -euo pipefail

: "${NSIGHT_PERF_BIN_PATH:?Set the directory containing the SDK service and tools}"
: "${AON_PYTHON:?Set the Python executable for the AON-enabled PyTorch build}"
: "${AON_EXAMPLE_SCRIPT:?Set the absolute path to aon_capture.py}"
output="${AON_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR}}/aon-${SLURM_JOB_ID}"
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
env LD_LIBRARY_PATH="${workload_ld}" "${AON_PYTHON}" "${AON_EXAMPLE_SCRIPT}" \
  --output "${output}/captures" --captures 2 --seconds 2
