# P0-2 · Add GSM8k Dataset

## Hypothesis
If AnswerableMath is near ceiling, GSM8k should replace it; otherwise GSM8k is
an additional math benchmark to test whether pruning's effect generalizes.

## Method
- **Saturation check** (data-driven, cached parquets, no GPU): computed the best
  single-sample average accuracy across all conditions on AnswerableMath.
- **Loader**: added `load_data_gsm8k` / `_parse_gsm8k_answer` / `gsm8k_preprocess`
  to [src/generate_model_answers.py](../../src/generate_model_answers.py),
  registered `gsm8k` in [src/probing_utils.py](../../src/probing_utils.py)
  (`LIST_OF_DATASETS`) and in [src/compute_correctness.py](../../src/compute_correctness.py)
  (`CORRECTNESS_FN['gsm8k'] = compute_correctness_math` — GSM8k's integer answers
  use the same numeric-substring matcher as MATH).
- **Unit test**: [src/test_gsm8k_dataset.py](../../src/test_gsm8k_dataset.py)
  (CPU, <60s; pure parse/preprocess/correctness + a network-gated real-loader smoke).
- **SLURM**: [scripts/slurm/run_generate_gsm8k_oscar.sh](../../scripts/slurm/run_generate_gsm8k_oscar.sh)
  — baseline/pruned generate + extract, all CLI-configurable (model, dataset, n).

## Results
**MATH is NOT saturated** → GSM8k is an **additional** dataset, not a replacement.

| Condition (AnswerableMath, single-sample avg acc) | Accuracy |
|---|---|
| Unpruned baseline | 0.758 |
| Best pruned config (p=5e-4, q=1e-6, one-word) | **0.793** (max over all configs) |
| Saturation threshold | 0.85 |

No condition exceeds 85%, so MATH stays in the suite and GSM8k is added alongside.

**GSM8k loader — validated (CPU, against the real source via AST isolation):**

| Check | Result |
|---|---|
| `_parse_gsm8k_answer` ("#### 18", "#### 1,234", "#### $5") | PASS |
| `gsm8k_preprocess` (instruct + base) | PASS |
| `compute_correctness_math` on GSM8k-style outputs [1,0,1] | PASS |
| Real loader download (HF `gsm8k`/`main`) | **PASS — 7473 train / 1319 test, label0=18.0** |

## Figures
None (loader/scaffold task).

## Interpretation
GSM8k integrates cleanly into the existing exact-match + best-of-N pipeline with
no new correctness logic (reuses `compute_correctness_math`). Unlike AnswerableMath
("Answer shortly."), GSM8k uses a step-by-step `gsm8k_preprocess` because GSM8k
needs multi-step reasoning; the numeric matcher scores the final number wherever it
appears, so CoT output is handled. Baseline/pruned accuracy numbers are **not yet
produced** — see blocker below.

## Next Steps
1. **Resolve the env blocker (below), then run** `run_generate_gsm8k_oscar.sh` for
   baseline + the best pruned model (p=1e-4, q=5e-6 — the confirmed best config
   from the `best_config_30_seeds` wandb project) and add GSM8k to the main table.
2. Add `gsm8k`/`gsm8k_test` to the array mapping in `submit_llmsknow_oscar.sh` if
   running through the full resample/probe pipeline rather than the standalone script.

## Failure Log
- **BLOCKER (env, not code): the `safety` conda env cannot import the LLMsKnow
  pipeline on Oscar.** It has `transformers==5.2.0` + `torch==2.10.0` (repo pins
  `4.42.3` / `2.2.0`) and was missing `baukit`. I installed `baukit` (`pip install
  --no-deps git+https://github.com/davidbau/baukit`), but its `__init__` requires
  `torchvision` (absent), and transformers 5.2's import graph itself pulls
  torchvision — so `import generate_model_answers` fails. Installing a torch-2.10-
  compatible torchvision (or downgrading transformers) is invasive and was left for
  the user. **The GSM8k code is nonetheless validated** by exec-ing the real
  function defs in isolation (AST extraction, bypassing the transformers import),
  and the loader successfully downloads/parses GSM8k. No new datasets were
  unavailable. Recommended fix: create a dedicated conda env from
  `requirements.txt` (pinned transformers 4.42.3 + torch 2.2.0 + baukit) for the
  LLMsKnow repo, separate from `safety`.
