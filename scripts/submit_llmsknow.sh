#!/bin/bash
# submit_llmsknow.sh
# Submits LLMsKnow pipeline phases as SLURM array jobs.
#
# Usage:
#   bash submit_llmsknow.sh --phase generate
#   bash submit_llmsknow.sh --phase extract
#   bash submit_llmsknow.sh --phase resample
#   bash submit_llmsknow.sh --phase extract_resamples
#   bash submit_llmsknow.sh --phase probe_all_layers    # single job, TriviaQA only
#   bash submit_llmsknow.sh --phase probe_choose --layer 15 --token last_q_token
#
# Flags:
#   --phase <name>           Pipeline phase to run (required)
#   --layer <N>              Layer index for probe_choose (default: 15)
#   --token <name>           Token position for probe_choose (default: last_q_token)
#   --throttle <N>           Max concurrent array tasks (default: 4); 0 = no limit
#   --array <range>          Override SLURM array range, e.g. --array 1,3 to re-run specific datasets
#
# Dataset index mapping (1-based):
#   1=triviaqa  2=triviaqa_test  3=popqa  4=popqa_test
#   5=pubqa     6=pubqa_test     7=math   8=math_test
#
# Each phase must complete before submitting the next.

BASE=/home/inguyen4/Desktop/research/interp/LLMsKnowPruningHallucinationBaseline
mkdir -p $BASE/logs

PHASE=""
LAYER=15
TOKEN="last_q_token"
THROTTLE=3
ARRAY_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)    PHASE="$2";          shift 2 ;;
        --layer)    LAYER="$2";          shift 2 ;;
        --token)    TOKEN="$2";          shift 2 ;;
        --throttle) THROTTLE="$2";       shift 2 ;;
        --array)    ARRAY_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$PHASE" ]]; then
    echo "ERROR: --phase is required"
    exit 1
fi

N_DATASETS=8

if [[ "$PHASE" == "probe_all_layers" ]]; then
    JOB=$(sbatch --parsable \
        --job-name="llmsknow_probe_all" \
        --output="$BASE/logs/probe_all_%j.out" \
        --error="$BASE/logs/probe_all_%j.err" \
        $BASE/scripts/run_llmsknow_probe_all.sh)
    echo "Submitted single job ${JOB} for phase: probe_all_layers (TriviaQA)"
    echo "After completion, check logs for best --layer and --token, then run:"
    echo "  bash submit_llmsknow.sh --phase probe_choose --layer <N> --token <TOKEN>"
else
    if [[ -n "$ARRAY_OVERRIDE" ]]; then
        ARRAY_RANGE="$ARRAY_OVERRIDE"
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
        --export=ALL,PHASE=${PHASE},LAYER=${LAYER},TOKEN=${TOKEN} \
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
fi
