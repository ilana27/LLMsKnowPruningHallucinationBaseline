#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --job-name=consolidate_scores
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o /home/inguyen4/Desktop/research/interp/LLMsKnowPruningHallucinationBaseline/logs/consolidate_scores_%j.out
#SBATCH -e /home/inguyen4/Desktop/research/interp/LLMsKnowPruningHallucinationBaseline/logs/consolidate_scores_%j.err

source /home/inguyen4/Desktop/research/interp/safety/.venv/bin/activate

cd ~/Desktop/research/interp/LLMsKnowPruningHallucinationBaseline/output

python - <<'EOF'
import torch, glob, os

chunk_dir = "llama-3.1-8b-instruct-chunks-triviaqa_test"
out_file  = "llama-3.1-8b-instruct-scores-triviaqa_test.pt"
tmp_file  = out_file + ".tmp"

all_scores, all_output_ids = [], []
for cf in sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.pt"))):
    print(f"Loading {cf}...", flush=True)
    chunk = torch.load(cf, weights_only=False)
    all_scores.extend(chunk["scores"])
    all_output_ids.extend(chunk["output_ids"])

print("Saving...", flush=True)
torch.save({"all_scores": all_scores, "all_output_ids": all_output_ids}, tmp_file)
os.rename(tmp_file, out_file)  # atomic replace so old file survives if it OOMs
print("Done.")
EOF
