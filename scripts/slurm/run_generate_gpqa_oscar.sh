#!/bin/bash
# run_generate_gpqa_oscar.sh
# Baseline (or pruned) GPQA Diamond multiple-choice evaluation on Oscar.
# GPQA is scored by letter exact-match (compute_correctness_gpqa) at generation
# time, so NO LLM extraction step is needed (unlike the short-answer QA datasets).
#
# Usage:
#   sbatch scripts/slurm/run_generate_gpqa_oscar.sh \
#       --model meta-llama/Llama-3.1-8B-Instruct
#   # pruned: point --model at a saved pruned-model dir.
# Random baseline accuracy = 25%.
#
#SBATCH --job-name=gen_gpqa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --output=/oscar/data/sbach/bats/projects/interp/LLMsKnowPruningHallucinationBaseline/logs/gen_gpqa_%j.out
#SBATCH --error=/oscar/data/sbach/bats/projects/interp/LLMsKnowPruningHallucinationBaseline/logs/gen_gpqa_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilana_nguyen1@brown.edu

set -euo pipefail

BASE=/oscar/data/sbach/bats/projects/interp/LLMsKnowPruningHallucinationBaseline
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="gpqa"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2";   shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$BASE/logs"

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate safety

cd "$BASE/src"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/oscar/scratch/inguyen4/wandb
export HF_DATASETS_CACHE=/oscar/scratch/inguyen4/hf_datasets
# GPQA Diamond is gated on HF; export HF_TOKEN before submitting if needed.
export PYTHONUNBUFFERED=1

echo "[gpqa] generate + letter-match: model=$MODEL dataset=$DATASET"
python generate_model_answers.py --model "$MODEL" --dataset "$DATASET"
echo "[gpqa] done."
