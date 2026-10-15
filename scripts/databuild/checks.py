"""Checks on a built data file that need no generation: prompt length and label leaks."""
import argparse
import unicodedata

from src.config import MODEL_REGISTRY
from src.data.loader import clean_answer

from .common import data_path, read_json

MAX_PROMPT_TOKENS = 256  # same limit as extract_features in src/data/feature_extractor.py


def answer_shape(text: str) -> str:
    """The form of a text without its meaning: A = capital, a = other letter, 0 = digit.

    Punctuation and spaces stay as they are.
    """
    return "".join("A" if c.isupper() else "0" if c.isdigit() else
                   "a" if unicodedata.category(c)[0] in "LM" else c for c in text)


def cmd_tokens(args: argparse.Namespace) -> None:
    """Count the prompts that the pipeline would cut at MAX_PROMPT_TOKENS."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY[args.model])
    rows = read_json(data_path(args.dataset, args.lang))
    prompts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows]
    lengths = sorted(len(ids) for ids in tokenizer(prompts)["input_ids"])
    cut = sum(n > MAX_PROMPT_TOKENS for n in lengths)
    print(f"{args.dataset} {args.lang} with {args.model}: {len(rows)} rows, "
          f"median {lengths[len(lengths) // 2]} tokens, longest {lengths[-1]}, "
          f"{cut} rows ({100 * cut / len(rows):.1f}%) over {MAX_PROMPT_TOKENS} tokens")
    if cut:
        print("The cut rows lose the end of the answer. Raise max_length in extract_features.")


def cmd_leak(args: argparse.Namespace) -> None:
    """AUC of a classifier that sees only the shape of the answer, not its meaning. No LLM.

    Answers are read as the pipeline reads them. Both rows of a question stay in the same fold.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    rows = read_json(data_path(args.dataset, args.lang))
    answers = [str(r["answer"]) for r in rows]
    if args.dataset == "triviaqa":
        answers = [clean_answer(a) for a in answers]  # as load_triviaqa does
    labels = [r["is_hallucination"] for r in rows]
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=False),
                          LogisticRegression(max_iter=2000))
    probs = cross_val_predict(model, [answer_shape(a) for a in answers], labels, cv=GroupKFold(5),
                              groups=[str(r["question"]) for r in rows], method="predict_proba")[:, 1]
    print(f"{args.dataset} {args.lang}: surface-only AUC {roc_auc_score(labels, probs):.3f} "
          "(0.5 = no leak; a pipeline AUC means little unless it is clearly above this)")
