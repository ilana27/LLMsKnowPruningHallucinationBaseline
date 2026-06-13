"""CPU unit test for the GSM8k dataset integration (<60s, no GPU).

Covers the pure logic (answer parsing, prompt building, numeric correctness) with
synthetic fixtures, plus an optional network-gated smoke test of the real loader.

Run:  python test_gsm8k_dataset.py     (or: pytest test_gsm8k_dataset.py)
"""
import math

from generate_model_answers import _parse_gsm8k_answer, gsm8k_preprocess, load_data_gsm8k
from compute_correctness import compute_correctness_math


def test_parse_gsm8k_answer():
    # GSM8k gold answers are reasoning text ending in "#### <number>"
    assert _parse_gsm8k_answer("Janet sells ... so #### 18") == 18.0
    assert _parse_gsm8k_answer("...\n#### 1,234") == 1234.0
    assert _parse_gsm8k_answer("#### $5") == 5.0
    assert _parse_gsm8k_answer("#### 72%") == 72.0
    assert math.isclose(_parse_gsm8k_answer("#### 3.5"), 3.5)


def test_gsm8k_preprocess_instruct_and_base():
    qs = ["What is 2+2?", "How many apples?"]
    instr = gsm8k_preprocess("meta-llama/Llama-3.1-8B-Instruct", qs, None)
    assert len(instr) == 2
    assert all(q in p for q, p in zip(qs, instr))
    assert "step by step" in instr[0].lower()
    base = gsm8k_preprocess("meta-llama/Meta-Llama-3-8B", qs, None)
    assert all(q in p for q, p in zip(qs, base))


def test_compute_correctness_numeric():
    # gsm8k reuses compute_correctness_math: gold number must appear in the output
    labels = [18.0, 72.0, 5.0]
    answers = [
        "After working it out the total is 18 dollars.",   # correct
        "I think the answer is 70.",                        # wrong (72 not present)
        "Step by step ... the final numeric answer is 5",   # correct
    ]
    out = compute_correctness_math(answers, labels)["correctness"]
    assert out == [1, 0, 1], out


def test_real_loader_smoke():
    """Network-gated: load 10 real GSM8k test examples and check the contract."""
    try:
        questions, labels = load_data_gsm8k(test=True)
    except Exception as e:  # offline / HF unavailable on this node
        print(f"SKIP test_real_loader_smoke (loader/network unavailable): {e}")
        return
    assert len(questions) == len(labels) and len(questions) > 100
    assert all(isinstance(float(l), float) for l in labels[:10])
    assert all(isinstance(q, str) and len(q) > 0 for q in questions[:10])
    print(f"OK real loader: {len(questions)} gsm8k_test examples, "
          f"example label={labels.iloc[0]}")


if __name__ == "__main__":
    fns = [test_parse_gsm8k_answer, test_gsm8k_preprocess_instruct_and_base,
           test_compute_correctness_numeric, test_real_loader_smoke]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if failed else 0)
