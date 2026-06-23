import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from transformers import set_seed

from compute_correctness import compute_correctness
from generate_model_answers import generate_model_answers
from probing_utils import load_model_and_validate_gpu, \
    LIST_OF_DATASETS, LIST_OF_MODELS, \
    MODEL_FRIENDLY_NAMES, LIST_OF_TEST_DATASETS


def parse_args_and_init_wandb():

    parser = argparse.ArgumentParser(
        description='Measure improvement rate across retries')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_resamples", default=30, type=int)
    parser.add_argument("--dataset", choices=LIST_OF_DATASETS+LIST_OF_TEST_DATASETS, required=True)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit_samples", type=int, required=False, default=None, help="Resample only a subset of the dataset")
    parser.add_argument("--tag", required=False, default="", type=str, help="Useful for running a few runs in parallel, see readme")

    args = parser.parse_args()

    model_short = MODEL_FRIENDLY_NAMES.get(args.model, args.model.split('/')[-1])
    wandb.init(
        project="retries_hallucinations",
        entity="ilananguyen-brown-university",
        name=f"{model_short}_{args.dataset}",
        config=vars(args)
    )

    return args

def main(args):
    model_output_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    data = pd.read_csv(model_output_file).reset_index(drop=True)

    if args.limit_samples is not None:
        data = data.sample(args.limit_samples, random_state=args.seed)
        sampled_indices = data.index.tolist()
        data = data.reset_index(drop=True)
    else:
        sampled_indices = None
    print(f"Length of data: {len(data)}")

    set_seed(args.seed)

    model, tokenizer = load_model_and_validate_gpu(args.model)

    stop_token_id = None
    if 'instruct' not in args.model.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]

    # math/gsm8k need more than the 100-token default or the multi-step reasoning
    # is truncated before the final answer (matches generate_model_answers.py).
    gen_max_new_tokens = 512 if any(k in args.dataset for k in ('math', 'gsm8k')) else 100

    greedy_correctness = np.mean(data['automatic_correctness'].mean())
    wandb.summary['greedy_correcntess'] = greedy_correctness

    all_textual_answers, all_input_output_ids, all_probe_predictions = [], [], []
    all_exact_answers = {
        "exact_answer": [],
        "valid_exact_answer": []
    }
    # --- per-round checkpointing: the textual_answers / input_output_ids files are
    # written (atomically) after each resample round and reloaded on restart, so a
    # killed job resumes from the last completed round instead of from scratch. ---
    out_dir = "../output/resampling"
    os.makedirs(out_dir, exist_ok=True)
    base = f"{out_dir}/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}"
    ta_path = f"{base}_textual_answers{args.tag}.pt"
    ioid_path = f"{base}_input_output_ids{args.tag}.pt"

    def _atomic_save(obj, path):
        tmp = f"{path}.tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    if os.path.exists(ta_path) and os.path.exists(ioid_path):
        try:
            ta_ckpt = torch.load(ta_path)
            ioid_ckpt = torch.load(ioid_path)
            n_done = min(len(ta_ckpt), len(ioid_ckpt))
            # Only resume if the checkpoint matches the current data size; guards against
            # stale checkpoints from a previous run with a different limit_samples
            # (e.g. the old 50-sample files).
            if 0 < n_done <= args.n_resamples and len(ta_ckpt[0]) == len(data):
                all_textual_answers = list(ta_ckpt[:n_done])
                all_input_output_ids = list(ioid_ckpt[:n_done])
                print(f"Resuming from checkpoint: {n_done}/{args.n_resamples} rounds done (data size {len(data)} matches)")
            else:
                stale = len(ta_ckpt[0]) if n_done else 'NA'
                print(f"Ignoring stale checkpoint (round size {stale} != data size {len(data)}); starting fresh")
        except Exception as e:
            print(f"Checkpoint load failed ({e}); starting fresh")

    for i in range(len(all_textual_answers), args.n_resamples):
        print(f"Retry #{i}")
        textual_answers, input_output_ids, _, _ = generate_model_answers(data.question, model,
                                                                                              tokenizer, model.device,
                                                                                              args.model,
                                                                                              do_sample=True,
                                                                                              output_scores=False,
                                                                                              temperature=args.temperature,
                                                                                              top_p=args.top_p,
                                                                                              max_new_tokens=gen_max_new_tokens,
                                                                                              stop_token_id=stop_token_id)
        all_textual_answers.append(textual_answers)
        all_input_output_ids.append(input_output_ids)
        _atomic_save(all_textual_answers, ta_path)
        _atomic_save(all_input_output_ids, ioid_path)
        print(f"Checkpoint saved: {len(all_textual_answers)}/{args.n_resamples} rounds")

    all_different_correctness = []
    total_correctness = np.zeros(len(all_textual_answers[0]))
    mean_correctness_after_k_retries = []
    for i in range(args.n_resamples):
        wrong_answers = data.incorrect_answer if ('incorrect_answer' in data) else None
        res = compute_correctness(data.question, args.dataset, args.model, data.correct_answer, model, all_textual_answers[i], tokenizer, wrong_answers)
        current_correctness = res['correctness']

        all_different_correctness.append(current_correctness)
        total_correctness = total_correctness + current_correctness
        mean_correctness = np.mean(total_correctness > 0)
        print("Total correctness:", mean_correctness)

        mean_correctness_after_k_retries.append(mean_correctness)
        wandb.log({'mean_correctness': mean_correctness, 'total_correctness': np.mean(total_correctness)})

        if 'exact_answer' in res:
            all_exact_answers['exact_answer'].append(res['exact_answer'])
            all_exact_answers['valid_exact_answer'].append([1] * len(res['exact_answer']))

    ax = sns.lineplot(x=range(0, args.n_resamples), y=mean_correctness_after_k_retries, marker='o')
    ax.set(xlabel='# retries', ylabel='Correctness')
    ax.set_xticks((1, args.n_resamples))
    wandb.summary['mean_correctness_after_k_retries'] = mean_correctness_after_k_retries

    ax.figure.savefig("temp_fig1.png", bbox_inches="tight")
    wandb.log({f"correctness": wandb.Image("temp_fig1.png")})
    os.remove("temp_fig1.png")

    if not os.path.exists("../output/resampling"):
        os.makedirs("../output/resampling")

    # textual_answers / input_output_ids were already written incrementally by the
    # per-round checkpointing above; final atomic re-save for consistency.
    _atomic_save(all_textual_answers, ta_path)
    _atomic_save(all_input_output_ids, ioid_path)
    if sampled_indices is not None:
        torch.save(sampled_indices, f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}_sample_indices{args.tag}.pt")

    if len(all_exact_answers['exact_answer']) != 0:
        all_exact_answers['exact_answer'] = [[all_exact_answers['exact_answer'][i][j] for i in range(0, args.n_resamples)] for j in
                                      range(0, len(data))]
        all_exact_answers['valid_exact_answer'] = [[all_exact_answers['valid_exact_answer'][i][j] for i in range(0, args.n_resamples)] for j in
                                      range(0, len(data))]
        torch.save(all_exact_answers, f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}_exact_answers{args.tag}.pt")

if __name__ == "__main__":
    args = parse_args_and_init_wandb()
    main(args)