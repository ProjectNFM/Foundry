#!/bin/bash
# Launch all 7 conditions for the POYO Overfitting Regularization experiment.
# Each condition is submitted as a SLURM job via Hydra's launcher (-m flag).
#
# Usage:
#   bash scripts/launch_overfit_regularization.sh          # submit to SLURM
#   bash scripts/launch_overfit_regularization.sh local    # run locally (debug)
#
# Conditions:
#   1. baseline    — WD=0.01, dropout=default(0.2/0.4/0.2), unfrozen
#   2. wd005       — WD=0.05
#   3. wd01        — WD=0.1
#   4. drop03      — all dropouts=0.3
#   5. drop05      — all dropouts=0.5
#   6. frozen_tok  — CWT-CNN frozen (cwt_lr_multiplier=0)
#   7. combined    — WD=0.1 + dropout=0.5 + frozen tokenizer

set -e

EXP="experiment=p300/brain_invaders_poyo_overfit_regularization"

if [[ "${1}" == "local" ]]; then
  MODE=""
  echo "Running LOCALLY (no SLURM). Pass no argument to submit to SLURM."
else
  MODE="-m"
  echo "Submitting 7 jobs to SLURM. Pass 'local' to run locally instead."
fi

# 1. Baseline
uv run python main.py $EXP \
  run.condition_tag=baseline \
  hyperparameters.weight_decay=0.01 \
  hyperparameters.cwt_lr_multiplier=1.0 $MODE

# 2. Weight decay 0.05
uv run python main.py $EXP \
  run.condition_tag=wd005 \
  hyperparameters.weight_decay=0.05 \
  hyperparameters.cwt_lr_multiplier=1.0 $MODE

# 3. Weight decay 0.1
uv run python main.py $EXP \
  run.condition_tag=wd01 \
  hyperparameters.weight_decay=0.1 \
  hyperparameters.cwt_lr_multiplier=1.0 $MODE

# 4. Dropout 0.3 (uniform across all layers)
uv run python main.py $EXP \
  run.condition_tag=drop03 \
  model.ffn_dropout=0.3 \
  model.lin_dropout=0.3 \
  model.atn_dropout=0.3 \
  hyperparameters.cwt_lr_multiplier=1.0 $MODE

# 5. Dropout 0.5 (uniform across all layers)
uv run python main.py $EXP \
  run.condition_tag=drop05 \
  model.ffn_dropout=0.5 \
  model.lin_dropout=0.5 \
  model.atn_dropout=0.5 \
  hyperparameters.cwt_lr_multiplier=1.0 $MODE

# 6. Frozen tokenizer (CWT-CNN lr=0)
uv run python main.py $EXP \
  run.condition_tag=frozen_tok \
  hyperparameters.cwt_lr_multiplier=0.0 $MODE

# 7. Combined (WD=0.1 + dropout=0.5 + frozen tokenizer)
uv run python main.py $EXP \
  run.condition_tag=combined \
  hyperparameters.weight_decay=0.1 \
  model.ffn_dropout=0.5 \
  model.lin_dropout=0.5 \
  model.atn_dropout=0.5 \
  hyperparameters.cwt_lr_multiplier=0.0 $MODE
