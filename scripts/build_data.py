"""Build the dataset files that src/data/loader.py reads from data/.

Run from the repository root, in this order:

  python scripts/build_data.py halueval-en
  python scripts/build_data.py triviaqa-en                              # Gemma-2-2B hard negatives
  python scripts/build_data.py check     --dataset halueval --lang am   # read the round trip first
  python scripts/build_data.py translate --dataset halueval --lang am
  python scripts/build_data.py tokens    --dataset halueval --lang am --model mistral-7b
  python scripts/build_data.py leak      --dataset triviaqa --lang en   # no model; run after every build

Languages:   fr (French), bn (Bangla), am (Amharic), vi (Vietnamese)
Translator:  google/translategemma-12b-it, loaded in 4-bit (--model-id and --quant change that)
Long steps save their progress next to the output file and resume after a restart.
The steps live in scripts/databuild/; this file only parses the command line.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_REGISTRY  # noqa: E402  (needs the repository root on sys.path)
from databuild.checks import cmd_leak, cmd_tokens  # noqa: E402
from databuild.english import cmd_halueval_en, cmd_triviaqa_en  # noqa: E402
from databuild.translation import LANGS, TRANSLATOR_MODEL, cmd_check, cmd_translate  # noqa: E402

ERROR_LOG = "logs/scripts_error.log"  # errors only; the full output is in the terminal
DATASETS = ["halueval", "triviaqa"]


def log_errors_to_file(path: str = ERROR_LOG) -> None:
    """Append the ERROR and CRITICAL records to path. A run without errors writes no file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8", delay=True)
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("halueval-en").set_defaults(func=cmd_halueval_en)

    p = sub.add_parser("triviaqa-en")
    p.add_argument("--pairs", type=int, default=5000, help="hard-negative pairs to keep (2 rows each)")
    p.add_argument("--questions", type=int, default=16000, help="questions to ask; about a third give a pair")
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=cmd_triviaqa_en)

    for name, func in (("translate", cmd_translate), ("check", cmd_check)):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=DATASETS)
        p.add_argument("--lang", required=True, choices=LANGS)
        p.add_argument("--model-id", default=TRANSLATOR_MODEL,
                       help="other TranslateGemma model, for example google/translategemma-4b-it")
        p.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"],
                       help="GPU only; the 12B model needs 4bit or 8bit on a 24 GB card. "
                            "4bit computes in bfloat16; 8bit is slower and computes in float16")
        p.add_argument("--batch-size", type=int, default=64,
                       help="texts per generation call. The time per call hardly grows with it, "
                            "so a bigger batch is faster until GPU memory runs out")
        if name == "check":
            p.add_argument("--n", type=int, default=100, help="rows to sample")
        p.set_defaults(func=func)

    for name, func in (("tokens", cmd_tokens), ("leak", cmd_leak)):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=DATASETS)
        p.add_argument("--lang", required=True, choices=["en", *LANGS])
        if name == "tokens":
            p.add_argument("--model", default="mistral-7b", choices=list(MODEL_REGISTRY))
        p.set_defaults(func=func)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_errors_to_file()
    try:
        args.func(args)
    except Exception:
        logging.critical("failed: %s", " ".join(sys.argv), exc_info=True)
        raise


if __name__ == "__main__":
    main()
