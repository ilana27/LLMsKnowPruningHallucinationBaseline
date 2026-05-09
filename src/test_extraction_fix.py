"""
Quick sanity-check for the split('\n')[0] fix in extract_exact_answer.

Tests two things:
  1. correctness==1 branch: substring search in the raw resample answers.
  2. correctness==0 branch: post-processing of a mocked Llama raw output,
     comparing the old [-1] vs new [0] split.

Run from the src/ directory:
    python test_extraction_fix.py
"""

# ---------------------------------------------------------------------------
# A few raw resample answers from the provided CSV row (Moon River question)
# ---------------------------------------------------------------------------
QUESTION = "What was the name of the Oscar-winning song 'performed' by Audrey Hepburn in `Breakfast at Tiffany's'?"
CORRECT_ANSWER = "['Moon River', 'Moon River (song)', 'Moon River Cha Cha']"
CORRECT_ANSWERS = ["Moon River", "Moon River (song)", "Moon River Cha Cha"]

RESAMPLE_ANSWERS = [
    "<|start_header_id|>assistant<|end_header_id|>\n\n\"Moon River\" is the Oscar-winning song...",
    "<|start_header_id|>assistant<|end_header_id|>\n\nThe song performed by Audrey Hepburn in the movie 'Breakfast at Tiffany\\'s' was called 'Moon River'.",
    "<|start_header_id|>assistant<|end_header_id|>\n\nI think there's been a mix-up. The song is 'Moon River,' written by Henry Mancini and Johnny Mercer.",
    # One that intentionally does NOT contain the answer (to test correctness==0 path)
    "<|start_header_id|>assistant<|end_header_id|>\n\nI cannot determine which specific song won an Oscar.",
]

# ---------------------------------------------------------------------------
# Mock Llama raw extraction outputs that the extraction model might produce.
# The model often keeps generating Q/A pairs after the actual answer.
# ---------------------------------------------------------------------------
MOCK_LLAMA_OUTPUTS = [
    # Clean single-line answer (both old and new work)
    "Moon River\n",
    # Answer followed by continued Q&A generation (the bug case)
    "Moon River\n\nQ: Which play by Oscar Wilde contains...\nA: The Importance of Being Earnest",
    # Answer with Exact answer: prefix not fully stripped, then junk
    "Exact answer: Moon River\n\nNote: The long answer mentioned My Fair Lady...",
    # Empty first line then answer — tricky case
    "\nMoon River",
    # NO ANSWER case
    "NO ANSWER",
]


# ---------------------------------------------------------------------------
# Post-processing (mirrors extract_exact_answer.py lines 113-115)
# ---------------------------------------------------------------------------
def postprocess_old(raw: str) -> str:
    return (raw
            .replace(".<|eot_id|>", "")
            .replace("<|eot_id|>", "")
            .replace("Exact answer:", "")
            .split('\n')[-1]   # BUG: last line
            .split("(")[0]
            .strip()
            .strip("."))


def postprocess_new(raw: str) -> str:
    lines = [l for l in raw
             .replace(".<|eot_id|>", "")
             .replace("<|eot_id|>", "")
             .replace("Exact answer:", "")
             .split('\n') if l.strip()]
    return lines[0].split("(")[0].strip().strip(".") if lines else ""


# ---------------------------------------------------------------------------
# Correctness==1 branch: substring search
# ---------------------------------------------------------------------------
def extract_correct_branch(model_answer: str, correct_answers: list) -> str:
    found_ans_index = len(model_answer)
    found_ans = ""
    for ans in correct_answers:
        idx = model_answer.lower().find(ans.lower())
        if idx != -1 and idx < found_ans_index:
            found_ans = ans
            found_ans_index = idx
    if found_ans_index == -1 or found_ans == "":
        return "NOT FOUND"
    return model_answer[found_ans_index: found_ans_index + len(found_ans)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TEST 1: correctness==1 branch (substring search on resample answers)")
    print("=" * 70)
    for i, raw in enumerate(RESAMPLE_ANSWERS):
        # Mimic what the script does: take first line of raw resample
        model_answer = raw.split("\n")[0]  # this strips the header line
        # Actually the full answer is used; take everything after the header
        # (header ends after the second \n\n)
        parts = raw.split("\n\n", 1)
        model_answer = parts[1] if len(parts) > 1 else raw

        result = extract_correct_branch(model_answer, CORRECT_ANSWERS)
        contains = any(a.lower() in raw.lower() for a in CORRECT_ANSWERS)
        correctness = 1 if contains else 0
        branch = "substring" if correctness == 1 else "LLM extraction needed"
        print(f"  Sample {i}: correctness={correctness} | branch={branch} | extracted='{result}'")

    print()
    print("=" * 70)
    print("TEST 2: correctness==0 branch — old vs new post-processing")
    print("=" * 70)
    all_passed = True
    for raw in MOCK_LLAMA_OUTPUTS:
        old = postprocess_old(raw)
        new = postprocess_new(raw)
        ok = new in ("Moon River", "NO ANSWER", "Moon River Cha Cha")
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  {status}  old='{old}'  →  new='{new}'")
        if old != new:
            print(f"        (raw first line: {repr(raw.split(chr(10))[0])})")

    print()
    print("All correctness==0 tests passed!" if all_passed else "SOME TESTS FAILED — check output above.")


if __name__ == "__main__":
    main()
