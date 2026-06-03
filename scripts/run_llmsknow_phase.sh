#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40|l40s|rtx_a6000"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilana_nguyen1@brown.edu
# Job name, output, and error paths are set by submit_llmsknow.sh via sbatch CLI flags.

BASE=/home/inguyen4/Desktop/research/interp/LLMsKnowPruningHallucinationBaseline

DATASETS=(triviaqa triviaqa_test
          popqa popqa_test pubqa pubqa_test math math_test)

if [ -n "${DATASET_OVERRIDE}" ]; then
    DATASET=${DATASET_OVERRIDE}
else
    DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID - 1))]}
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} phase=${PHASE}"

source /home/inguyen4/Desktop/research/interp/safety/.venv/bin/activate

cd $BASE/src

export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/cs/data/bats/users/inguyen4/wandb

MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_FRIENDLY=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')

# Resolve a --limit_samples value: "full" (or empty) => row count of the
# dataset's answers CSV; an integer passes through unchanged. Keeping the value
# equal to the full size makes resampling.py's sample(N, seed) a deterministic
# full permutation, which extract_exact_answer.py and probe_choose_answer.py
# both reproduce with the same seed (42), so every row stays aligned.
resolve_limit() {
    local req="$1" ds="$2"
    if [[ "$req" == "full" || -z "$req" ]]; then
        local csv="$BASE/output/${MODEL_FRIENDLY}-answers-${ds}.csv"
        if [[ ! -f "$csv" ]]; then
            echo "ERROR: answers CSV not found for ${ds}: ${csv}" >&2
            exit 1
        fi
        echo $(( $(wc -l < "$csv") - 1 ))  # subtract header row
    else
        echo "$req"
    fi
}

case "$PHASE" in
  generate)
    python generate_model_answers.py \
      --model $MODEL \
      --dataset $DATASET ;;
  extract)
    python extract_exact_answer.py \
      --model $MODEL \
      --extraction_model mistralai/Mistral-7B-Instruct-v0.2 \
      --dataset $DATASET ;;
  resample)
    LIM=$(resolve_limit "$LIMIT_SAMPLES" "$DATASET") || exit 1
    echo "Using limit_samples=${LIM} for ${DATASET}"
    python resampling.py \
      --model $MODEL \
      --dataset $DATASET \
      --n_resamples 30 \
      --seed 42 \
      --limit_samples ${LIM} ;;
  resample_parallel)
    # 8 datasets × 6 tags = 48 tasks (task IDs 1-48).
    # Seeds avoid 42 (used by the serial resample phase).
    N_TAGS=6
    DATASET_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / N_TAGS ))
    TAG=$(( (SLURM_ARRAY_TASK_ID - 1) % N_TAGS + 1 ))
    DATASET=${DATASETS[$DATASET_IDX]}
    SEEDS=(0 5 26 63 70 100)
    SEED=${SEEDS[$((TAG - 1))]}
    LIM=$(resolve_limit "$LIMIT_SAMPLES" "$DATASET") || exit 1
    # NOTE: each tag uses a different seed, so sample(N, seed) permutes questions
    # differently per tag. resampling_merge_runs.py stacks tags by position, so
    # the merged file is only row-aligned if all tags share question order. Prefer
    # the serial `resample` phase (seed 42) for the full-data run.
    echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} tag=${TAG} seed=${SEED} limit_samples=${LIM}"
    python resampling.py \
      --model $MODEL \
      --dataset $DATASET \
      --n_resamples 5 \
      --seed $SEED \
      --tag $TAG \
      --limit_samples ${LIM} ;;
  extract_resamples)
    RLIM=$(resolve_limit "$RESAMPLE_LIMIT_SAMPLES" "$DATASET") || exit 1
    echo "Using resample_limit_samples=${RLIM} for ${DATASET}"
    python extract_exact_answer.py \
      --model $MODEL \
      --extraction_model mistralai/Mistral-7B-Instruct-v0.2 \
      --dataset $DATASET \
      --do_resampling 30 \
      --resample_seed 42 \
      --resample_limit_samples ${RLIM} ;;
  extract_resamples_parallel)
    # 8 tasks (1-8), chunk size is dynamic (resample_limit_samples / n_chunks). Requires --dataset flag in submit script.
    # If N is not divisible by n_chunks the tail (N % n_chunks rows) is dropped; probe_choose handles the
    # shorter length via iloc[:n_extracted], so it stays aligned.
    RLIM=$(resolve_limit "$RESAMPLE_LIMIT_SAMPLES" "$DATASET") || exit 1
    echo "Using resample_limit_samples=${RLIM} for ${DATASET}"
    python extract_exact_answer.py \
      --model $MODEL \
      --extraction_model mistralai/Mistral-7B-Instruct-v0.2 \
      --dataset $DATASET \
      --do_resampling 30 \
      --resample_seed 42 \
      --resample_limit_samples ${RLIM} \
      --tag $SLURM_ARRAY_TASK_ID \
      --n_chunks 8 ;;
  probe_all_layers)
    python probe_all_layers_and_tokens.py \
      --model $MODEL \
      --dataset $DATASET \
      --seed 0 \
      --n_samples 1000 \
      --probe_at mlp_last_layer_only_input ;;
  probe_train)
    python probe.py \
      --model $MODEL \
      --dataset $DATASET \
      --layer $LAYER \
      --token $TOKEN \
      --probe_at mlp \
      --save_clf \
      --seeds 0 ;;
  probe_choose)
    python probe_choose_answer.py \
      --model $MODEL \
      --dataset $DATASET \
      --layer $LAYER \
      --token $TOKEN \
      --probe_at mlp \
      --seeds 0 5 26 42 63 ;;
  *)
    echo "Unknown phase: $PHASE"; exit 1 ;;
esac
