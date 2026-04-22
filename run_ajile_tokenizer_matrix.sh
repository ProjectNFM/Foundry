#!/usr/bin/env bash
set -euo pipefail

TOKENIZERS=(
  per_channel_per_timepoint_linear
  spatial_session_per_timepoint_identity
  spatial_session_cwt
)

WANDB_GROUP="AJILE12_INTRASESSION_TOKENIZER_MATRIX"
EXPERIMENT="tokenizer_explore/poyo_ajile_tokenizer_matrix"
JOB_PIDS=()

stop_all_jobs() {
  if ((${#JOB_PIDS[@]} == 0)); then
    return
  fi

  echo "Stopping ${#JOB_PIDS[@]} running jobs..."
  for pid in "${JOB_PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done

  wait || true
  exit 130
}

trap stop_all_jobs INT TERM

run_behavior_job() {
  local gpu_id="$1"
  local tokenizer="$2"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    uv run python main.py \
    experiment="${EXPERIMENT}" \
    model/tokenizer="${tokenizer}" \
    data.task_type=behavior \
    model.zero_output_timestamps=true \
    trainer.callbacks.early_stopping.monitor=val/ajile_active_behavior_f1 \
    trainer.callbacks.early_stopping.mode=max \
    trainer.callbacks.early_stopping.patience=20 \
    trainer.callbacks.model_checkpoint.monitor=val/ajile_active_behavior_f1 \
    trainer.callbacks.model_checkpoint.mode=max \
    trainer.callbacks.model_checkpoint.auto_insert_metric_name=false \
    trainer.callbacks.model_checkpoint.filename=best-behavior \
    run.group="${WANDB_GROUP}" \
    run.name="ajile_behavior_${tokenizer}" &

  JOB_PIDS+=("$!")
}

run_pose_job() {
  local gpu_id="$1"
  local tokenizer="$2"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    uv run python main.py \
    experiment="${EXPERIMENT}" \
    model/tokenizer="${tokenizer}" \
    module=regression \
    data.task_type=pose_estimation \
    model.zero_output_timestamps=false \
    trainer.callbacks.early_stopping.monitor=val/ajile_pose_estimation_r2 \
    trainer.callbacks.early_stopping.mode=max \
    trainer.callbacks.early_stopping.patience=20 \
    trainer.callbacks.model_checkpoint.monitor=val/ajile_pose_estimation_r2 \
    trainer.callbacks.model_checkpoint.mode=max \
    trainer.callbacks.model_checkpoint.auto_insert_metric_name=false \
    trainer.callbacks.model_checkpoint.filename=best-pose \
    run.group="${WANDB_GROUP}" \
    run.name="ajile_pose_${tokenizer}" &

  JOB_PIDS+=("$!")
}

for idx in "${!TOKENIZERS[@]}"; do
  run_behavior_job "${idx}" "${TOKENIZERS[$idx]}"
done

for idx in "${!TOKENIZERS[@]}"; do
  run_pose_job "$((idx + 3))" "${TOKENIZERS[$idx]}"
done

exit_code=0
for pid in "${JOB_PIDS[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

exit "${exit_code}"
