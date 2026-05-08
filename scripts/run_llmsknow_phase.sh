#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40|l40s|rtx_a6000"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilana_nguyen1@brown.edu
# Job name, output, and error paths are set by submit_llmsknow.sh via sbatch CLI flags.

BASE=/home/inguyen4/Desktop/research/interp/LLMsKnowPruningHallucinationBaseline

DATASETS=(triviaqa triviaqa_test
          popqa popqa_test pubqa pubqa_test math math_test)

DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} phase=${PHASE}"

source /home/inguyen4/Desktop/research/interp/safety/.venv/bin/activate

cd $BASE/src

export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/cs/data/bats/users/inguyen4/wandb

MODEL=meta-llama/Llama-3.1-8B-Instruct

case "$PHASE" in
  generate)
    python generate_model_answers.py \
      --model $MODEL \
      --dataset $DATASET ;;
  extract)
    python extract_exact_answer.py \
      --model $MODEL \
      --dataset $DATASET ;;
  resample)
    python resampling.py \
      --model $MODEL \
      --dataset $DATASET \
      --n_resamples 30 \
      --seed 42 \
      --limit_samples 200 ;;
  extract_resamples)
    python extract_exact_answer.py \
      --model $MODEL \
      --dataset $DATASET \
      --do_resampling 30 ;;
  probe_all_layers)
    python probe_all_layers_and_tokens.py \
      --model $MODEL \
      --dataset $DATASET \
      --seed 0 \
      --probe_at mlp_last_layer_only_input ;;
  probe_choose)
    python probe_choose_answer.py \
      --model $MODEL \
      --dataset $DATASET \
      --layer $LAYER \
      --token $TOKEN \
      --probe_at mlp ;;
  *)
    echo "Unknown phase: $PHASE"; exit 1 ;;
esac
