import argparse

import torch

from probing_utils import LIST_OF_MODELS, LIST_OF_DATASETS, LIST_OF_TEST_DATASETS, MODEL_FRIENDLY_NAMES

parser = argparse.ArgumentParser(description='Merge extract_resamples_parallel chunk runs')
parser.add_argument("--model", choices=LIST_OF_MODELS)
parser.add_argument("--dataset", choices=LIST_OF_DATASETS + LIST_OF_TEST_DATASETS, required=True)
parser.add_argument("--n_chunks", type=int, default=8)

args = parser.parse_args()

model = MODEL_FRIENDLY_NAMES[args.model]
dataset = args.dataset

exact_answers = []
valid_lst = []

for i in range(1, args.n_chunks + 1):
    path = f"../output/resampling/{model}_{dataset}_30_exact_answers{i}.pt"
    print("loaded", path)
    data = torch.load(path)
    exact_answers.extend(data["exact_answer"])
    valid_lst.extend(data["valid_exact_answer"])

print(f"Total samples merged: {len(exact_answers)}")
torch.save(
    {"exact_answer": exact_answers, "valid_exact_answer": valid_lst},
    f"../output/resampling/{model}_{dataset}_30_exact_answers.pt"
)
