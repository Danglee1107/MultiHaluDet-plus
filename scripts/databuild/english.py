"""The English source files: HaluEval as the loader builds it, TriviaQA with Gemma-2-2B hard negatives."""
import argparse
import os
import random
import re
import string

from src.data.loader import clean_answer, load_halueval

from .common import SEED, Row, data_path, map_resumable, write_json
from .generation import chat_generate, model_kwargs

NEGATIVES_MODEL = "google/gemma-2-2b-it"


def normalize_answer(text: str) -> str:
    """Lower case, no punctuation, no articles, single spaces: the form answers are compared in."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def is_wrong_answer(prediction: str, aliases: list[str]) -> bool:
    """True when the prediction matches no accepted answer, also not as a part of one."""
    pred = normalize_answer(prediction)
    if not pred:
        return False
    return not any(a and (a in pred or pred in a) for a in map(normalize_answer, aliases))


def build_hard_negative_rows(questions: list[str], aliases: list[list[str]], predictions: list[str],
                             pairs: int) -> list[Row]:
    """One correct row and one model-written wrong row per question that the model got wrong."""
    rows: list[Row] = []
    for question, accepted, prediction in zip(questions, aliases, predictions, strict=True):
        # Both answers in the same form, so that the translator also gets them in the same form.
        answer = clean_answer(prediction.splitlines()[0]) if prediction.strip() else ""
        if accepted and len(answer.split()) <= 12 and is_wrong_answer(answer, accepted):
            rows.append({"question": question, "answer": clean_answer(accepted[0]), "is_hallucination": 0})
            rows.append({"question": question, "answer": answer, "is_hallucination": 1})
        if len(rows) >= 2 * pairs:
            break
    return rows


def cmd_halueval_en(args: argparse.Namespace) -> None:
    out_path = data_path("halueval", "en")
    rows = load_halueval("en")
    write_json(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def cmd_triviaqa_en(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # These answers are in lower case and Gemma writes capitals: build_hard_negative_rows puts
    # both through clean_answer, or the case alone gives the label away (AUC 0.998).
    dataset = load_dataset("lucadiliello/triviaqa", split="train").select_columns(["question", "answers"])
    dataset = dataset.shuffle(seed=SEED).select(range(min(args.questions, len(dataset))))
    questions, aliases = list(dataset["question"]), list(dataset["answers"])

    tokenizer = AutoTokenizer.from_pretrained(NEGATIVES_MODEL)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(NEGATIVES_MODEL, **model_kwargs("none")).eval()

    def answer(batch: list[str]) -> list[str]:
        conversations = [[{"role": "user", "content":
                           f"Answer the question with a short answer only, no explanation.\n\nQuestion: {q}"}]
                         for q in batch]
        return chat_generate(model, tokenizer, conversations, max_new_tokens=32)

    out_path = data_path("triviaqa", "en")
    partial_path = out_path + ".partial"
    predictions = map_resumable(questions, answer, partial_path, args.batch_size)
    rows = build_hard_negative_rows(questions, aliases, predictions, args.pairs)
    random.Random(SEED).shuffle(rows)
    write_json(out_path, rows)
    os.remove(partial_path)
    print(f"{len(rows) // 2} hard negatives from {len(questions)} questions "
          f"({100 * len(rows) / 2 / len(questions):.0f}% answered wrong). Wrote {out_path}")
    if len(rows) < 2 * args.pairs:
        print(f"Fewer than the {args.pairs} pairs asked for: raise --questions")
