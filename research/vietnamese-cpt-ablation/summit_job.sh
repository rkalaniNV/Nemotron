#!/usr/bin/env bash

if [[ -n "${ZSH_VERSION:-}" ]]; then
  SCRIPT_SOURCE="${(%):-%N}"
else
  SCRIPT_SOURCE="${BASH_SOURCE[0]}"
fi
if [[ "$SCRIPT_SOURCE" != /* ]]; then
  SCRIPT_SOURCE="$PWD/$SCRIPT_SOURCE"
fi
if ! SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)" ||
   ! REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"; then
  echo "failed to resolve the Nemotron repository root" >&2
  return 1 2>/dev/null || exit 1
fi
cd "$REPO_ROOT" || {
  echo "failed to enter the Nemotron repository root: $REPO_ROOT" >&2
  return 1 2>/dev/null || exit 1
}

export PRETRAIN_BLEND_PATH="/mnt/lustre-shared/hvnguyen/data/processed/nemotron_3_nano_30b/stage0_pretrain/blend.json"
export PRETRAIN_OUTPUT_DIR="/mnt/lustre-shared/hvnguyen/checkpoints/vi-cpt-runtime-sweep"

A100_ENV_FILE="$PWD/research/vietnamese-cpt-ablation/envs/a100.toml"
H100_ENV_FILE="$PWD/research/vietnamese-cpt-ablation/envs/h100.toml"
A100_PROFILE="lepton_nano_cpt_a100"
H100_PROFILE="lepton_nano_cpt_h100_gcp_tcpxo"
H100_4NODE_PROFILE="lepton_nano_cpt_h100_gcp_4node"
H100_NCCL_DEBUG_PROFILE="lepton_nano_cpt_h100_gcp_nccl_debug"
CONFIG_ROOT="research/vietnamese-cpt-ablation/configs"

nemotron_cli() {
  if command -v uv >/dev/null 2>&1; then
    uv run nemotron "$@"
  elif [[ -x "$PWD/.venv/bin/nemotron" ]]; then
    "$PWD/.venv/bin/nemotron" "$@"
  else
    echo "nemotron CLI not found: install uv or create $PWD/.venv" >&2
    return 127
  fi
}

config_for_case() {
  local case="$1"
  local config_path

  config_path="$(find "$CONFIG_ROOT" -type f -name "${case}.yaml" -print)"
  if [[ -z "$config_path" ]]; then
    echo "config not found under $CONFIG_ROOT: $case" >&2
    return 2
  fi
  if [[ "$config_path" == *$'\n'* ]]; then
    echo "duplicate config name under $CONFIG_ROOT: $case" >&2
    return 2
  fi
  printf '%s\n' "$config_path"
}

run_case() {
  local case="$1"
  local job_name="${2:-${case#vi_cpt_}}"
  local profile="${3:-$A100_PROFILE}"
  local env_file="${4:-$A100_ENV_FILE}"
  local config_path

  config_path="$(config_for_case "$case")" || return

  NEMOTRON_ENV_FILE="$env_file" nemotron_cli steps run pretrain/megatron_bridge \
    -c "$config_path" \
    -b "$profile" \
    "run.env.job_name=${job_name}" \
    "run.env.env_vars.PRETRAIN_BLEND_PATH=${PRETRAIN_BLEND_PATH}" \
    "run.env.env_vars.PRETRAIN_OUTPUT_DIR=${PRETRAIN_OUTPUT_DIR}"
}

dry_run_case() {
  local case="$1"
  local job_name="${2:-${case#vi_cpt_}}"
  local profile="${3:-$A100_PROFILE}"
  local env_file="${4:-$A100_ENV_FILE}"
  local config_path

  config_path="$(config_for_case "$case")" || return

  NEMOTRON_ENV_FILE="$env_file" nemotron_cli steps run pretrain/megatron_bridge \
    -c "$config_path" \
    --run "$profile" \
    --dry-run \
    "run.env.job_name=${job_name}" \
    "run.env.env_vars.PRETRAIN_BLEND_PATH=${PRETRAIN_BLEND_PATH}" \
    "run.env.env_vars.PRETRAIN_OUTPUT_DIR=${PRETRAIN_OUTPUT_DIR}"
}

run_h100_case() {
  local case="$1"
  local job_name="${2:-${case#vi_cpt_}}"
  run_case "$case" "$job_name" "$H100_PROFILE" "$H100_ENV_FILE"
}

dry_run_h100_case() {
  local case="$1"
  local job_name="${2:-${case#vi_cpt_}}"
  dry_run_case "$case" "$job_name" "$H100_PROFILE" "$H100_ENV_FILE"
}

run_h100_4node_case() {
  local case="$1"
  local job_name="${2:-32g-${case#vi_cpt_}}"
  run_case "$case" "$job_name" "$H100_4NODE_PROFILE" "$H100_ENV_FILE"
}

dry_run_hardware_baseline_a100() {
  dry_run_case \
    vi_cpt_31_a100_hardware_baseline \
    vi-hw-31-a100-baseline \
    "$A100_PROFILE" \
    "$A100_ENV_FILE"
}

run_hardware_baseline_a100() {
  run_case \
    vi_cpt_31_a100_hardware_baseline \
    vi-hw-31-a100-baseline \
    "$A100_PROFILE" \
    "$A100_ENV_FILE"
}

dry_run_hardware_baseline_h100_gcp() {
  dry_run_case \
    vi_cpt_32_h100_gcp_hardware_baseline \
    vi-h100-32-gcp-baseline \
    "$H100_PROFILE" \
    "$H100_ENV_FILE"
}

run_hardware_baseline_h100_gcp() {
  run_case \
    vi_cpt_32_h100_gcp_hardware_baseline \
    vi-h100-32-gcp-baseline \
    "$H100_PROFILE" \
    "$H100_ENV_FILE"
}

dry_run_h100_nccl_diagnostic() {
  dry_run_case \
    vi_cpt_33_h100_nccl_diagnostic \
    vi-h100-33-nccl-debug \
    "$H100_NCCL_DEBUG_PROFILE" \
    "$H100_ENV_FILE"
}

run_h100_nccl_diagnostic() {
  run_case \
    vi_cpt_33_h100_nccl_diagnostic \
    vi-h100-33-nccl-debug \
    "$H100_NCCL_DEBUG_PROFILE" \
    "$H100_ENV_FILE"
}


# run_h100_case vi_cpt_21_h100_mbs2_selective_alltoall          vi-h100-21-a2a-mbs2
# run_h100_case vi_cpt_22_h100_mbs2_selective_alltoall_tp_overlap vi-h100-22-a2a-tpo
# run_h100_case vi_cpt_23_h100_mbs2_selective_deepep              vi-h100-23-deepep
# run_h100_case vi_cpt_24_h100_mbs2_selective_deepep_tp_overlap   vi-h100-24-deepep-tpo
# run_h100_case vi_cpt_25_h100_mbs4_selective_deepep_tp_overlap   vi-h100-25-deepep-tpo-mbs4
# run_h100_case vi_cpt_26_h100_mbs2_selective_deepep_tp_overlap_fp8 vi-h100-26-fp8-deepep-tpo
# run_h100_case vi_cpt_27_h100_mbs4_selective_deepep_tp_overlap_fp8 vi-h100-27-fp8-deepep-mbs4
# run_h100_case vi_cpt_28_h100_mbs2_selective_deepep_tp_overlap_shared vi-h100-28-deepep-tpo-shared
# run_h100_case vi_cpt_29_h100_mbs2_no_recompute_deepep_tp_overlap vi-h100-29-no-recompute
# run_h100_case vi_cpt_30_h100_tp2_mbs2_selective_deepep_tp_overlap vi-h100-30-tp2-deepep-tpo