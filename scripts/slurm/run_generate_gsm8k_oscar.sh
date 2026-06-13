#!/bin/bash
# run_generate_gsm8k_oscar.sh
# Baseline (or pruned) GSM8k generation + exact-answer extraction on Oscar.
# All configurable values are CLI args (no hardcoded model/dataset).
#
# Usage:
#   sbatch scripts/slurm/run_generate_gsm8k_oscar.sh \
#       --model meta-llama/Llama-3.1-8B-Instruct --dataset gsm8k_test --n-samples 1319
#   # pruned: point --model at a saved pruned-model dir produced by hallucination_prune.py
#
#SBATCH --job-name=gen_gsm8k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --output=/oscar/data/sbach/bats/projects/interp/LLMsKnowPruningHallucinationBaseline/logs/gen_gsm8k_%j.out
#SBATCH --error=/oscar/data/sbach/bats/projects/interp/LLMsKnowPruningHallucinationBaseline/logs/gen_gsm8k_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilana_nguyen1@brown.edu

set -euo pipefail

BASE=/oscar/data/sbach/bats/projects/interp/LLMsKnowPruningHallucinationBaseline
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="gsm8k_test"
N_SAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2";     shift 2 ;;
    --dataset)   DATASET="$2";   shift 2 ;;   # gsm8k | gsm8k_test
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
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
export PYTHONUNBUFFERED=1

NS_ARG=""
[[ -n "$N_SAMPLES" ]] && NS_ARG="--n_samples $N_SAMPLES"

echo "[gsm8k] generate: model=$MODEL dataset=$DATASET $NS_ARG"
python generate_model_answers.py --model "$MODEL" --dataset "$DATASET" $NS_ARG

echo "[gsm8k] extract exact answers"
python extract_exact_answer.py --model "$MODEL" --extraction_model "$MODEL" --dataset "$DATASET"

echo "[gsm8k] done. Answers CSV: $BASE/output/$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')-answers-${DATASET}.csv"
