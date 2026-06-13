"""CPU unit test for the GPQA Diamond multiple-choice integration (<60s, no GPU).

Run:  python test_gpqa_dataset.py     (or: pytest test_gpqa_dataset.py)
"""
from generate_model_answers import load_data_gpqa, gpqa_preprocess
from compute_correctness import compute_correctness_gpqa


def test_compute_correctness_gpqa():
    answers = [
        "The answer is C.",          # -> C
        "B",                          # -> B
        "I'd go with A) because ...", # -> A
        "Answer: D",                  # -> D
        "I'm not sure.",              # -> no letter
    ]
    labels = ["C", "A", "A", "D", "B"]
    out = compute_correctness_gpqa(answers, labels)["correctness"]
    assert out == [1, 0, 1, 1, 0], out


def test_gpqa_preprocess():
    qs = ["Q1\n\nA. x\nB. y\nC. z\nD. w"]
    instr = gpqa_preprocess("meta-llama/Llama-3.1-8B-Instruct", qs, None)
    assert "only the letter" in instr[0].lower()
    assert qs[0] in instr[0]


def test_real_loader_smoke():
    """Network-gated: GPQA Diamond is gated on HF; skip cleanly if unavailable."""
    try:
        q1, l1 = load_data_gpqa()
        q2, l2 = load_data_gpqa()  # determinism
    except Exception as e:
        print(f"SKIP test_real_loader_smoke (gated/network): {e}")
        return
    assert len(q1) == len(l1) == 198, len(q1)
    assert set(l1.unique()) <= {"A", "B", "C", "D"}
    assert list(l1) == list(l2), "option shuffle must be deterministic"
    # each question embeds all four lettered options
    assert all(all(f"{c}." in q for c in "ABCD") for q in q1.iloc[:5])
    print(f"OK gpqa loader: 198 MCQs, label dist={l1.value_counts().to_dict()}")


if __name__ == "__main__":
    fns = [test_compute_correctness_gpqa, test_gpqa_preprocess, test_real_loader_smoke]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if failed else 0)
