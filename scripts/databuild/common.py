"""File helpers shared by every build step: paths, atomic JSON writes, resumable batch loops."""
import json
import os
from collections.abc import Callable

from tqdm import tqdm

SEED = 42

Row = dict[str, str | int]
BatchFn = Callable[[list[str]], list[str]]


def data_path(dataset: str, lang: str) -> str:
    return f"data/{dataset}_{lang}.json"


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)  # atomic: a kill during the write keeps the old file


def map_resumable(items: list[str], fn: BatchFn, partial_path: str,
                  batch_size: int, save_every: int = 20) -> list[str]:
    """Apply fn to batches of items. Finished outputs are kept in partial_path for a restart."""
    outputs: list[str] = []
    if os.path.exists(partial_path):
        saved = read_json(partial_path)
        if saved["inputs"] == items[:len(saved["outputs"])]:
            outputs = saved["outputs"]
            print(f"Resuming from item {len(outputs)}/{len(items)}")
        else:
            print(f"Ignoring {partial_path}: it belongs to different input")

    def save() -> None:
        write_json(partial_path, {"inputs": items[:len(outputs)], "outputs": outputs})

    for n, i in enumerate(tqdm(range(len(outputs), len(items), batch_size))):
        batch = items[i:i + batch_size]
        batch_out = fn(batch)
        if len(batch_out) != len(batch):
            raise RuntimeError(f"batch at item {i}: got {len(batch_out)} outputs for {len(batch)} inputs")
        outputs.extend(batch_out)
        if n % save_every == save_every - 1:
            save()
    save()
    return outputs
