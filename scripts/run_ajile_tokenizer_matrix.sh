#!/usr/bin/env bash
set -euo pipefail

TOKENIZERS=(
  per_channel_per_timepoint_linear
  per_channel_cwt
  per_channel_resample_cnn
  spatial_session_per_timepoint_identity
  spatial_session_cwt
  spatial_session_resample_cnn
)

EFFECTIVE_BATCH_SIZE=1024
MAX_BATCH_SIZES=(
  64
  128
  128
  512
  512
  512
)

WANDB_GROUP="AJILE12_INTRASESSION_TOKENIZER_MATRIX"
EXPERIMENT="tokenizer_explore/poyo_ajile_tokenizer_matrix"
JOB_PIDS=()

accum_steps() {
  local effective="$1"
  local batch="$2"

  if ((batch <= 0)); then
    echo "ERROR: batch size must be > 0, got ${batch}" >&2
    exit 1
  fi

  if ((effective % batch != 0)); then
    echo "ERROR: effective batch size ${effective} not divisible by ${batch}" >&2
    exit 1
  fi

  echo $((effective / batch))
}

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
  local batch_size="$3"
  local accumulate_grad_batches="$4"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    uv run python main.py \
    experiment="${EXPERIMENT}" \
    model/tokenizer="${tokenizer}" \
    hyperparameters.batch_size="${batch_size}" \
    trainer.accumulate_grad_batches="${accumulate_grad_batches}" \
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
  local batch_size="$3"
  local accumulate_grad_batches="$4"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    uv run python main.py \
    experiment="${EXPERIMENT}" \
    model/tokenizer="${tokenizer}" \
    hyperparameters.batch_size="${batch_size}" \
    trainer.accumulate_grad_batches="${accumulate_grad_batches}" \
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

if ((${#TOKENIZERS[@]} != ${#MAX_BATCH_SIZES[@]})); then
  echo "ERROR: TOKENIZERS and MAX_BATCH_SIZES must have the same length" >&2
  exit 1
fi

for idx in "${!TOKENIZERS[@]}"; do
  batch_size="${MAX_BATCH_SIZES[$idx]}"
  accumulate_grad_batches="$(accum_steps "${EFFECTIVE_BATCH_SIZE}" "${batch_size}")"
  run_behavior_job "${idx}" "${TOKENIZERS[$idx]}" "${batch_size}" "${accumulate_grad_batches}"
done

for idx in "${!TOKENIZERS[@]}"; do
  batch_size="${MAX_BATCH_SIZES[$idx]}"
  accumulate_grad_batches="$(accum_steps "${EFFECTIVE_BATCH_SIZE}" "${batch_size}")"
  run_pose_job "$((idx + 4))" "${TOKENIZERS[$idx]}" "${batch_size}" "${accumulate_grad_batches}"
done

exit_code=0
for pid in "${JOB_PIDS[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

exit "${exit_code}"
