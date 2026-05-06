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

BASE=/users/inguyen4/data/bats/projects/interp/LLMsKnowPruningHallucinationBaseline

source /home/inguyen4/Desktop/research/interp/safety/.venv/bin/activate

cd $BASE/src

export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/cs/data/bats/users/inguyen4/wandb

python probe_all_layers_and_tokens.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset triviaqa \
    --probe_at mlp_last_layer_only
