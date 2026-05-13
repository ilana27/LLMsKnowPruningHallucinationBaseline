import argparse
import sys

import numpy as np
import wandb
from sklearn.utils import resample

from compute_correctness import compute_correctness_triviaqa, compute_correctness_math, compute_correctness

sys.path.append("../src")

import pandas as pd
import torch
from tqdm import tqdm

from probing_utils import load_model_and_validate_gpu, tokenize, generate, LIST_OF_MODELS, MODEL_FRIENDLY_NAMES, \
    LIST_OF_TEST_DATASETS, LIST_OF_DATASETS


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=LIST_OF_DATASETS + LIST_OF_TEST_DATASETS)
    parser.add_argument("--do_resampling", type=int, required=False, default=0, help="If 0, the script will extract exact answers from the model answers. If > 0, the script will extract exact answers from the resampled model answers (looking for a file of do_resampling resamples).")
    parser.add_argument("--get_extraction_stats", action='store_true', default=False, help="Purely for getting statistics. If activated, the file will not be saved.")
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--extraction_model", choices=LIST_OF_MODELS, default='mistralai/Mistral-7B-Instruct-v0.2', help="model used for exact answer extraction")
    parser.add_argument("--model", choices=LIST_OF_MODELS, default='mistralai/Mistral-7B-Instruct-v0.2', help="model which answers are to be extracted")
    parser.add_argument("--resample_seed", type=int, default=None, help="Seed used in resampling.py --seed; needed to recover sampled indices when --limit_samples was used.")
    parser.add_argument("--resample_limit_samples", type=int, default=None, help="Value of --limit_samples passed to resampling.py; used to recover which rows were sampled.")
    parser.add_argument("--tag", type=str, default="", help="Chunk tag (1-8) for parallel extraction runs; appended to output filename.")
    parser.add_argument("--n_chunks", type=int, default=0, help="Total number of chunks; 0 means no chunking.")

    args = parser.parse_args()
    model_short = MODEL_FRIENDLY_NAMES.get(args.model, args.model.split('/')[-1])
    run_name = f"{model_short}_{args.dataset}" + (f"_chunk{args.tag}" if args.tag else "")
    wandb.init(
        project="extract_exact_answer",
        name=run_name,
        config=vars(args)
        )

    return args


def extract_exact_answer(model, tokenizer, correctness, question, model_answer, correct_answer, model_name):

    if correctness == 1:
        found_ans_index = len(model_answer)
        found_ans = ""

        try:
            correct_answer_ = eval(correct_answer)
            if type(correct_answer_) == list:
                correct_answer = correct_answer_
        except:
            correct_answer = correct_answer

        if type(correct_answer) == list:
            for ans in correct_answer:
                ans_index = model_answer.lower().find(ans.lower())
                if ans_index != -1 and ans_index < found_ans_index:
                    found_ans = ans
                    found_ans_index = ans_index
        elif type(correct_answer) in [int, float]:
            found_ans_index = model_answer.lower().find(str(round(correct_answer)))
            found_ans = str(round(correct_answer))
            if found_ans_index == -1:
                found_ans_index = model_answer.lower().find(str(correct_answer))
                found_ans = str(correct_answer)
        else:
            found_ans_index = model_answer.lower().find(correct_answer.lower())
            found_ans = correct_answer


        if found_ans_index == -1:
            print("##")
            print(model_answer)
            print("##")
            print(correct_answer)
            print("ERROR!", question)
        exact_tokens = list(range(found_ans_index, found_ans_index + len(found_ans)))
        exact_answer = "".join([model_answer[i] for i in exact_tokens])
        valid = 1
    else:
        prompt = f"""
        Extract from the following long answer the short answer, only the relevant tokens. If the long answer does not answer the question, output NO ANSWER.

        Q: Which musical featured the song The Street Where You Live?
        A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady." It is one of the most famous songs from the show, and it is sung by Professor Henry Higgins as he reflects on the transformation of Eliza Doolittle and the memories they have shared together.
        Exact answer: My Fair Lady

        Q: Which Swedish actress won the Best Supporting Actress Oscar for Murder on the Orient Express?
        A: I'm glad you asked about a Swedish actress who won an Oscar for "Murder on the Orient Express," but I must clarify that there seems to be a misunderstanding here. No Swedish actress has won an Oscar for Best Supporting Actress for that film. The 1974 "Murder on the Orient Express" was an American production, and the cast was predominantly British and American. If you have any other questions or if there's another
        Exact answer: NO ANSWER

        Q: {question}
        A: {model_answer}
        Exact answer:
        """
        model_input = tokenize(prompt, tokenizer, model_name).to(model.device)
        valid = 0
        retries = 0
        sample = True
        print("###")
        while valid == 0 and retries < 5:
            with torch.no_grad():
                model_output = generate(model_input, model, model_name, sample, False)
                exact_answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])
            if 'mistral' in model_name.lower():
                exact_answer = exact_answer.replace(".</s>", "").replace("</s>", "").split('\n')[0].split("(")[
                    0].strip().strip(".")
            elif 'llama' in model_name.lower():
                lines = [l for l in exact_answer.replace(".<|eot_id|>", "").replace("<|eot_id|>", "").replace("Exact answer:", "").split('\n') if l.strip()]
                exact_answer = (lines[0].split("(")[0].strip().strip(".") if lines else "")
            else:
                print("Model is not supported. Exisitng...")
                exit(1)

            if type(model_answer) == float:
                exact_answer = "NO ANSWER"
                valid = 0
            elif exact_answer.lower() in model_answer.lower():
                valid = 1
            elif exact_answer == "NO ANSWER":
                valid = 1
            retries += 1

    return exact_answer, valid


def main():
    args = parse_args()
    model, tokenizer = load_model_and_validate_gpu(args.extraction_model)
    source_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    resampling_file = f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.do_resampling}_textual_answers.pt"
    if args.do_resampling > 0:
        destination_file = f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.do_resampling}_exact_answers{args.tag}.pt"
    else:
        destination_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"

    model_answers = pd.read_csv(source_file)
    print(f"Length of data: {len(model_answers)}")

    if args.do_resampling > 0:
        all_resample_answers = torch.load(resampling_file)
        n_resampled = len(all_resample_answers[0])
        if len(model_answers) != n_resampled:
            assert args.resample_seed is not None and args.resample_limit_samples is not None, (
                "Resampling was run with --limit_samples; pass --resample_seed and "
                "--resample_limit_samples to recover the sampled subset."
            )
            sampled_indices = model_answers.sample(
                args.resample_limit_samples, random_state=args.resample_seed
            ).index.tolist()
            model_answers = model_answers.loc[sampled_indices].reset_index(drop=True)

    if args.n_chunks > 0 and args.tag:
        chunk_idx = int(args.tag) - 1  # tags are 1-indexed
        chunk_size = len(model_answers) // args.n_chunks
        start = chunk_idx * chunk_size
        end = start + chunk_size
        model_answers = model_answers.iloc[start:end].reset_index(drop=True)
        if args.do_resampling > 0:
            all_resample_answers = [answers[start:end] for answers in all_resample_answers]

    exact_answers = []
    valid_lst = []
    ctr = 0
    ctr_no_answer = 0

    if args.n_samples > 0:
        model_answers = resample(model_answers, n_samples=args.n_samples, stratify=model_answers['automatic_correctness'])

    for idx, row in tqdm(model_answers.iterrows()):
        print(f"###### sample {idx} #######")

        if 'raw_question' in row:
            question_col = 'raw_question'
        else:
            question_col = 'question'

        if args.do_resampling <= 0:
            if ('natural_questions' in source_file) or args.get_extraction_stats:
                automatic_correctness = 0
            else:
                automatic_correctness = row['automatic_correctness']

            if 'instruct' not in args.model.lower():
                model_answer = row['model_answer'].split("\n")[0]
            else:
                model_answer = row['model_answer']

            exact_answer, valid = extract_exact_answer(model, tokenizer,
                                                       automatic_correctness,
                                                       row[question_col], model_answer,
                                                       row['correct_answer'], args.extraction_model)
            exact_answers.append(exact_answer)
            valid_lst.append(valid)
            if exact_answer == 'NO ANSWER':
                ctr_no_answer += 1
            if valid == 1:
                ctr += 1
        else:
            exact_answers_specific_index = []
            valid_lst_specific_index = []
            for resample_answers in all_resample_answers:
                assert(len(model_answers) == len(resample_answers))
                resample_answer = resample_answers[idx].split("\n")[0]

                automatic_correctness = compute_correctness([row.question], args.dataset, args.model, [row['correct_answer']], model, [resample_answer], tokenizer, None)['correctness'][0]

                exact_answer, valid = extract_exact_answer(model, tokenizer,
                                                           automatic_correctness,
                                                           row[question_col], resample_answer,
                                                           row['correct_answer'], args.model)
                exact_answers_specific_index.append(exact_answer)
                valid_lst_specific_index.append(valid)
                if exact_answer == 'NO ANSWER':
                    ctr_no_answer += 1
                if valid == 1:
                    ctr += 1
            exact_answers.append(exact_answers_specific_index)
            valid_lst.append(valid_lst_specific_index)

    if args.do_resampling > 0:
        total_n_answers = len(model_answers) * len(all_resample_answers)
    else:
        total_n_answers = len(model_answers)

    wandb.summary['successful_extractions'] = ctr / total_n_answers
    wandb.summary['no_answer'] = ctr_no_answer / total_n_answers

    if not args.get_extraction_stats:
        if args.do_resampling <= 0:
            model_answers['exact_answer'] = exact_answers
            model_answers['valid_exact_answer'] = valid_lst
            model_answers.to_csv(destination_file)
        else:
            torch.save({
                "exact_answer": exact_answers,
                "valid_exact_answer": valid_lst
            }, destination_file)
    else:
        model_answers['exact_answer'] = exact_answers
        model_answers['valid_exact_answer'] = valid_lst

if __name__ == "__main__":
    main()
