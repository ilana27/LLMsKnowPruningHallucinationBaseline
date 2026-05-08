#!/bin/bash
# submit_llmsknow.sh
# Submits LLMsKnow pipeline phases as SLURM array jobs.
#
# Usage:
#   bash submit_llmsknow.sh --phase generate
#   bash submit_llmsknow.sh --phase extract
#   bash submit_llmsknow.sh --phase resample
#   bash submit_llmsknow.sh --phase resample_parallel   # 48-task array (8 datasets × 6 tags)
#   bash submit_llmsknow.sh --phase extract_resamples
#   bash submit_llmsknow.sh --phase probe_all_layers    # defaults to train splits 1,3,5,7
#   bash submit_llmsknow.sh --phase probe_choose --layer 15 --token last_q_token
#
# triviaqa + popqa train, 8000 samples
# bash scripts/submit_llmsknow.sh --phase resample_parallel --array 1-6,13-18 --limit_samples 8000

# triviaqa_test + popqa_test, 800 samples
# bash scripts/submit_llmsknow.sh --phase resample_parallel --array 7-12,19-24 --limit_samples 800
# Flags:
#   --phase <name>           Pipeline phase to run (required)
#   --layer <N>              Layer index for probe_choose (default: 15)
#   --token <name>           Token position for probe_choose (default: last_q_token)
#   --throttle <N>           Max concurrent array tasks (default: 0 = no limit)
#   --array <range>          Override SLURM array range, e.g. --array 1,3 to re-run specific datasets
#
# Dataset index mapping (1-based, standard phases):
#   1=triviaqa  2=triviaqa_test  3=popqa  4=popqa_test
#   5=pubqa     6=pubqa_test     7=math   8=math_test
#
# resample_parallel task IDs (dataset × tag, tags 1-6, seeds 0 5 26 63 70 100):
#   triviaqa: 1-6   triviaqa_test: 7-12   popqa: 13-18   popqa_test: 19-24
#   pubqa: 25-30    pubqa_test: 31-36     math: 37-42    math_test: 43-48
#
# Each phase must complete before submitting the next.
# After resample_parallel finishes, run merge manually (no GPU needed):
#   cd src && for ds in triviaqa triviaqa_test popqa popqa_test pubqa pubqa_test math math_test; do
#     python resampling_merge_runs.py --model meta-llama/Llama-3.1-8B-Instruct --dataset $ds; done

BASE=/home/inguyen4/Desktop/research/interp/LLMsKnowPruningHallucinationBaseline
mkdir -p $BASE/logs

PHASE=""
LAYER=15
TOKEN="last_q_token"
THROTTLE=0
ARRAY_OVERRIDE=""
LIMIT_SAMPLES=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)          PHASE="$2";          shift 2 ;;
        --layer)          LAYER="$2";          shift 2 ;;
        --token)          TOKEN="$2";          shift 2 ;;
        --throttle)       THROTTLE="$2";       shift 2 ;;
        --array)          ARRAY_OVERRIDE="$2"; shift 2 ;;
        --limit_samples)  LIMIT_SAMPLES="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$PHASE" ]]; then
    echo "ERROR: --phase is required"
    exit 1
fi

N_DATASETS=8

if [[ -n "$ARRAY_OVERRIDE" ]]; then
    ARRAY_RANGE="$ARRAY_OVERRIDE"
elif [[ "$PHASE" == "probe_all_layers" ]]; then
    ARRAY_RANGE="1,3,5,7"  # train splits only
elif [[ "$PHASE" == "resample_parallel" ]]; then
    ARRAY_RANGE="1-$((N_DATASETS * 6))"  # 48 tasks: 8 datasets × 6 tags
else
    ARRAY_RANGE="1-${N_DATASETS}"
fi

if [[ $THROTTLE -gt 0 ]]; then
    ARRAY_RANGE="${ARRAY_RANGE}%${THROTTLE}"
fi

JOB=$(sbatch --parsable \
    --job-name="llmsknow_${PHASE}" \
    --output="$BASE/logs/${PHASE}_%A_%a.out" \
    --error="$BASE/logs/${PHASE}_%A_%a.err" \
    --array=${ARRAY_RANGE} \
    --export=ALL,PHASE=${PHASE},LAYER=${LAYER},TOKEN=${TOKEN},LIMIT_SAMPLES=${LIMIT_SAMPLES} \
    $BASE/scripts/run_llmsknow_phase.sh)

echo "Submitted array job ${JOB} for phase: ${PHASE} (array: ${ARRAY_RANGE})"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Dataset mapping:"
DATASETS=(triviaqa triviaqa_test
          popqa popqa_test pubqa pubqa_test math math_test)
for (( i=0; i<N_DATASETS; i++ )); do
    echo "  Task $((i+1)): ${DATASETS[$i]}"
done
