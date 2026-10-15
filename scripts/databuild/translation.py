"""Translation with TranslateGemma: the model, the check for bad outputs, and the two commands."""
import argparse
import csv
import os
import random
import unicodedata
from collections import Counter
from collections.abc import Callable

from tqdm import tqdm

from .common import SEED, BatchFn, Row, data_path, map_resumable, read_json, write_json
from .generation import chat_generate, model_kwargs, split_on_oom

TRANSLATOR_MODEL = "google/translategemma-12b-it"
# Unicode name prefix of each target script. The keys are also the TranslateGemma language codes.
SCRIPTS = {"fr": "LATIN", "bn": "BENGALI", "am": "ETHIOPIC", "vi": "LATIN"}
LANGS = list(SCRIPTS)

TranslateFn = Callable[[list[str], str, str], list[str]]


def load_translator(model_id: str, quant: str) -> TranslateFn:
    """Return translate(texts, source_lang, target_lang) for a TranslateGemma model."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"  # generation continues from the last real token
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs(quant)).eval()

    def translate(texts: list[str], src: str, tgt: str) -> list[str]:
        conversations = [[{"role": "user", "content": [{
            "type": "text", "source_lang_code": src, "target_lang_code": tgt, "text": text,
        }]}] for text in texts]
        # A short input sometimes makes the model explain or repeat until the token limit,
        # which also holds up the whole batch: bound the output by the input length.
        longest = max(len(ids) for ids in processor.tokenizer(texts)["input_ids"])
        return chat_generate(model, processor, conversations,
                             max_new_tokens=min(256, 32 + 3 * longest))

    return translate


def suspect_reason(source: str, output: str, lang: str) -> str:
    """Why output is clearly not a translation of source into lang; "" when it looks fine."""
    if not output:
        return "empty"
    if "\n" in output:
        return "several lines"
    if "�" in output:
        return "broken character"
    if len(output) > 3 * len(source) + 20:
        return "too long"
    script = SCRIPTS[lang]
    names = [unicodedata.name(c, "") for c in output if c.isalpha() and c not in source]
    if any(not name.startswith(("LATIN", script)) for name in names):
        return "wrong script"
    if len(source.split()) >= 4:  # a sentence, not a name or a number, which may stay as it is
        letters = [c for c in output if c.isalpha()]
        in_target = sum(unicodedata.name(c, "").startswith(script) for c in letters)
        if output == source or (script != "LATIN" and 2 * in_target < len(letters)):
            return "not in target language"
    return ""


def retry_suspects(table: dict[str, str], translate: BatchFn, lang: str,
                   log_path: str | None, batch_size: int = 8) -> None:
    """Translate suspect entries of table once more in small batches; keep English if still suspect.

    Every suspect text is written to log_path with both outputs, so the cause can be read later.
    """
    reasons = {text: reason for text, output in table.items()
               if (reason := suspect_reason(text, output, lang))}
    suspects = list(reasons)
    log = []
    for i in tqdm(range(0, len(suspects), batch_size), desc="retry suspect"):
        batch = suspects[i:i + batch_size]
        for text, retry in zip(batch, translate(batch), strict=True):
            fixed = not suspect_reason(text, retry, lang)
            log.append({"english": text, "reason": reasons[text], "first": table[text],
                        "retry": retry, "kept": "retry" if fixed else "english"})
            table[text] = retry if fixed else text
    kept_english = sum(entry["kept"] == "english" for entry in log)
    print(f"{len(suspects)} of {len(table)} texts ({100 * len(suspects) / max(1, len(table)):.1f}%) "
          f"had a suspect translation: {dict(Counter(reasons.values()))}. "
          f"The retry fixed {len(suspects) - kept_english}; {kept_english} keep their English text.")
    if log_path:
        write_json(log_path, log)
        print(f"Suspect texts with both outputs: {log_path}")


def translate_rows(rows: list[Row], translate: BatchFn, partial_path: str, batch_size: int,
                   lang: str, log_path: str | None = None) -> list[Row]:
    """Translate question and answer of every row; labels are copied unchanged."""
    # ponytail: each text is translated alone, so a one-word answer has no question as context.
    # Translate question and answer in one prompt if short answers come out wrong.
    # Unique texts, longest first: similar lengths share a batch (little padding), and the batch
    # that needs the most GPU memory runs first, so a too-large --batch-size shows at once.
    texts = sorted({str(r[k]) for r in rows for k in ("question", "answer")} - {""},
                   key=lambda t: (-len(t), t))
    translate = split_on_oom(translate)
    table = dict(zip(texts, map_resumable(texts, translate, partial_path, batch_size), strict=True))
    retry_suspects(table, translate, lang, log_path)
    table[""] = ""
    return [{"question": table[str(r["question"])], "answer": table[str(r["answer"])],
             "is_hallucination": r["is_hallucination"]} for r in rows]


def cmd_translate(args: argparse.Namespace) -> None:
    rows = read_json(data_path(args.dataset, "en"))
    translate = load_translator(args.model_id, args.quant)
    out_path = data_path(args.dataset, args.lang)
    partial_path = f"{out_path}.translategemma.partial"
    translated = translate_rows(rows, lambda texts: translate(texts, "en", args.lang),
                                partial_path, args.batch_size, args.lang,
                                log_path=out_path.removesuffix(".json") + ".suspect.json")
    write_json(out_path, translated)
    os.remove(partial_path)
    print(f"Wrote {len(translated)} rows to {out_path}")


def cmd_check(args: argparse.Namespace) -> None:
    """Translate a sample to the target language and back, for reading against the English."""
    rows = random.Random(SEED).sample(read_json(data_path(args.dataset, "en")), args.n)
    translate = load_translator(args.model_id, args.quant)
    out_path = f"data/check_{args.dataset}_{args.lang}_translategemma.tsv"
    n_suspect = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["field", "suspect", "english", args.lang, "back_to_english"])
        for field in ("question", "answer"):
            english = [str(r[field]) for r in rows]
            forward: list[str] = []
            back: list[str] = []
            for i in tqdm(range(0, len(english), args.batch_size), desc=field):
                batch_forward = translate(english[i:i + args.batch_size], "en", args.lang)
                forward += batch_forward
                back += translate(batch_forward, args.lang, "en")
            flags = [suspect_reason(e, f, args.lang) for e, f in zip(english, forward, strict=True)]
            n_suspect += sum(bool(flag) for flag in flags)
            writer.writerows(zip([field] * len(english), flags, english, forward, back, strict=True))
    print(f"Wrote {out_path}. Read it: 'english' and 'back_to_english' must mean the same.")
    print(f"{n_suspect} of {2 * len(rows)} translations are suspect (runaway text or wrong script).")
