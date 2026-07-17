#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:05:00

set -a
source .env
set +a
export RANK=0
uv run python -m foundry.tools.stage_data --experiment pretraining/poyo_pretrain_tokenizer_sweep

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
srun uv run "$@"
