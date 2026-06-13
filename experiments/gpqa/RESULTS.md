# P0-3a · PhD-level STEM Skill (GPQA Diamond)

## Hypothesis
If pruning's hallucination-reduction effect generalizes to hard reasoning, it
should not degrade (and may help) accuracy on GPQA Diamond — though at 8B the
absolute accuracy will be low (random = 25%).

## Method
- Added `load_data_gpqa` + `gpqa_preprocess` to
  [src/generate_model_answers.py](../../src/generate_model_answers.py) and
  `compute_correctness_gpqa` to [src/compute_correctness.py](../../src/compute_correctness.py),
  registered `gpqa` in `LIST_OF_DATASETS` and `CORRECTNESS_FN`.
- **Format**: HF `Idavidrein/gpqa` / `gpqa_diamond` (198 PhD-level MCQs). For each
  item the 4 options (1 correct + 3 incorrect) are shuffled with a **deterministic
  per-example seed** (`np.random.RandomState(i)`); the question embeds the A/B/C/D
  block and the label is the correct letter.
- **Scoring**: letter exact-match at generation time (prefer an explicit
  "answer: X" pattern, else first standalone A–D token). No LLM extraction step.
- **Unit test**: [src/test_gpqa_dataset.py](../../src/test_gpqa_dataset.py) (CPU, <60s).
- **SLURM**: [scripts/slurm/run_generate_gpqa_oscar.sh](../../scripts/slurm/run_generate_gpqa_oscar.sh).

## Results
Loader + scoring **validated** (CPU, against real source via AST isolation):

| Check | Result |
|---|---|
| `compute_correctness_gpqa` (5 cases incl. "Answer: D", bare "B") → [1,0,1,1,0] | PASS |
| `gpqa_preprocess` (letter instruction) | PASS |
| Real loader (gpqa_diamond) | **PASS — 198 MCQs** |
| Deterministic option shuffle (two loads identical) | PASS |
| Label balance | A:58 B:43 C:57 D:40 (not degenerate) |

Baseline/pruned accuracy numbers **not yet produced** — blocked on the env issue
(see GSM8k RESULTS Failure Log; same `safety`-env transformers/torch/baukit problem).

## Figures
None (loader/scaffold task).

## Interpretation
GPQA fits the pipeline as a generation task with a custom letter-match correctness
fn — no probing/extraction changes needed. The deterministic shuffle makes
baseline-vs-pruned comparisons reproducible and position-bias-controlled. Whether
pruning helps/hurts on PhD-level STEM is the open empirical question, answerable
once the env is fixed and the SLURM script is run for baseline + best pruned config
(p=1e-4, q=5e-6).

## Next Steps
1. Fix env, run `run_generate_gpqa_oscar.sh` for baseline + pruned, report accuracy
   vs the 25% random floor, add to the main table.
2. If accuracy ≈ random for both, GPQA may be too hard at 8B to discriminate the
   pruning effect — consider reporting it as a generalization-bound rather than a
   headline result.

## Failure Log
- GPQA Diamond is gated on HF; it downloaded fine here (HF token already present in
  `~/.netrc`/cache). If a fresh node lacks access, export `HF_TOKEN` before submit.
- Same env blocker as P0-2 prevents an end-to-end run; logic validated in isolation.
