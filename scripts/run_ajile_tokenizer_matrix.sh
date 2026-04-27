#!/usr/bin/env bash
set -euo pipefail

# ── Task: "behavior" or "pose_estimation" ────────────────────────────────────
TASK="pose_estimation"

# ── Tokenizers (4 entries → 8 runs: fold 0 & fold 1 each) ───────────────────
TOKENIZERS=(
  per_channel_cwt
  per_channel_cwt_concat
  spatial_session_cwt
  spatial_session_cwt_common
)

EFFECTIVE_BATCH_SIZE=1024
MAX_BATCH_SIZES=(
  64
  128
  512
  512
)

FOLDS=(0 1)

WANDB_GROUP="AJILE12_INTRASESSION_TOKENIZER_CWT"
EXPERIMENT="tokenizer_explore/poyo_ajile_tokenizer_matrix"
JOB_PIDS=()

# ── Task-specific overrides ──────────────────────────────────────────────────
case "${TASK}" in
  behavior)
    MONITOR_METRIC="val/ajile_active_behavior_auroc"
    TASK_OVERRIDES=(
      data.task_type=behavior
      model.zero_output_timestamps=true
      "trainer.callbacks.model_checkpoint.filename=best-behavior"
    )
    ;;
  pose_estimation)
    MONITOR_METRIC="val/ajile_pose_estimation_r2"
    TASK_OVERRIDES=(
      module=regression
      data.task_type=pose_estimation
      model.zero_output_timestamps=false
      "trainer.callbacks.model_checkpoint.filename=best-pose"
    )
    ;;
  *)
    echo "ERROR: unknown TASK '${TASK}' (expected 'behavior' or 'pose_estimation')" >&2
    exit 1
    ;;
esac

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

if ((${#TOKENIZERS[@]} != ${#MAX_BATCH_SIZES[@]})); then
  echo "ERROR: TOKENIZERS and MAX_BATCH_SIZES must have the same length" >&2
  exit 1
fi

gpu_id=0
for idx in "${!TOKENIZERS[@]}"; do
  tokenizer="${TOKENIZERS[$idx]}"
  batch_size="${MAX_BATCH_SIZES[$idx]}"
  accumulate_grad_batches="$(accum_steps "${EFFECTIVE_BATCH_SIZE}" "${batch_size}")"

  for fold in "${FOLDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      uv run python main.py \
      experiment="${EXPERIMENT}" \
      model/tokenizer="${tokenizer}" \
      hyperparameters.batch_size="${batch_size}" \
      hyperparameters.fold_number="${fold}" \
      trainer.accumulate_grad_batches="${accumulate_grad_batches}" \
      trainer.callbacks.early_stopping.monitor="${MONITOR_METRIC}" \
      trainer.callbacks.early_stopping.mode=max \
      trainer.callbacks.early_stopping.patience=20 \
      trainer.callbacks.model_checkpoint.monitor="${MONITOR_METRIC}" \
      trainer.callbacks.model_checkpoint.mode=max \
      trainer.callbacks.model_checkpoint.auto_insert_metric_name=false \
      "${TASK_OVERRIDES[@]}" \
      run.group="${WANDB_GROUP}" \
      run.name="ajile_${TASK}_${tokenizer}_fold${fold}" &

    JOB_PIDS+=("$!")
    gpu_id=$((gpu_id + 1))
  done
done

exit_code=0
for pid in "${JOB_PIDS[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

exit "${exit_code}"
