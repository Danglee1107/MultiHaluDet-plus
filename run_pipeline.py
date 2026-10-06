import os

# Must be set before torch initialises CUDA; reduces memory fragmentation on small GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
import random
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import get_config, MODEL_REGISTRY, PIPELINE_MODELS
from src.data.loader import load_halueval, load_triviaqa
from src.data.feature_extractor import extract_dataset, SEQ_BASE_DIM, GLOB_BASE_DIM
from src.training.trainer import train_deep_model_fold, extract_features_batch, extract_logits_batch
from src.ensemble.meta_learner import get_ensemble, calibrate_ensemble, temperature_scale
from src.utils.metrics import find_best_thresholds, evaluate_all, compute_uncertainty_metrics
from src.utils.visualization import plot_publication_results


def _key(args, tag=True):
    """File-name suffix. --tag only applies to run-specific outputs (OOF, plots), never to the
    extracted LLM features, so one extraction can serve many seeds / ablations."""
    base = f"{args.dataset}_{args.lang}_{args.model}"
    return f"{base}_{args.tag}" if tag and args.tag else base


def stage_1_extract(args, config):
    print(f"\n--- Stage 1: Dynamic Layer Probing & Feature Extraction ---")
    print(f"Dataset: {args.dataset} ({args.lang}), Model: {args.model}")

    if args.dataset == "halueval":
        samples = load_halueval(lang=args.lang)
    else:
        samples = load_triviaqa(lang=args.lang, seed=config.seed)

    config.model_name = MODEL_REGISTRY[args.model]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        # 4-bit NF4: Mistral-7B takes ~4 GB and fits entirely on a 6 GB GPU.
        # device_map={"": 0} forces everything onto GPU 0 (no silent CPU/disk offload).
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            torch_dtype=config.model_dtype,
            device_map={"": 0},
        )
    else:
        model_llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.model_dtype,
        )
    model_llm.eval()

    key = _key(args, tag=False)
    os.makedirs('results/features', exist_ok=True)
    ckpt_path = f'results/features/partial_{key}.npz'
    X_seq, X_glob, y = extract_dataset(samples, tokenizer, model_llm, config, ckpt_path,
                                       batch_size=args.extract_batch_size)
    X_seq = np.nan_to_num(X_seq, nan=0.0)
    X_glob = np.nan_to_num(X_glob, nan=0.0)

    np.save(f'results/features/X_seq_{key}.npy', X_seq)
    np.save(f'results/features/X_glob_{key}.npy', X_glob)
    np.save(f'results/features/y_{key}.npy', y)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    del model_llm
    torch.cuda.empty_cache()
    gc.collect()
    print("Stage 1 completed. Features saved.")


def _train_fold(fold, tr_idx, val_idx, X_seq, X_glob, y, X_seq_test, X_glob_test, config):
    """Train one inner fold; return its out-of-fold and test outputs.

    Runs in its own process when config.fold_workers > 1, so it seeds itself.
    """
    seed = config.seed + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.fold_workers > 1 and not torch.cuda.is_available():
        # CPU only: share the cores between the fold processes.
        torch.set_num_threads(max(1, torch.get_num_threads() // config.fold_workers))
    model, best_auc = train_deep_model_fold(
        X_seq[tr_idx], X_glob[tr_idx], y[tr_idx],
        X_seq[val_idx], X_glob[val_idx], y[val_idx],
        config, log_prefix=f"[fold {fold + 1}] "
    )
    return (val_idx,
            extract_features_batch(model, X_seq[val_idx], X_glob[val_idx]),
            extract_features_batch(model, X_seq_test, X_glob_test),
            extract_logits_batch(model, X_seq[val_idx], X_glob[val_idx]),
            extract_logits_batch(model, X_seq_test, X_glob_test),
            best_auc)


def stage_2_3_train_oof(args, config):
    print(f"\n--- Stage 2 & 3: Multi-Scale Modeling & Out-of-Fold Generation ---")
    print(f"seed={config.seed} mixup={config.use_mixup} cutmix={config.use_cutmix} "
          f"layer_delta={config.use_layer_delta} layer_scale={config.scale_layer_weights} "
          f"rank_weight={config.rank_loss_weight}")
    feat_key = _key(args, tag=False)
    try:
        X_seq = np.load(f'results/features/X_seq_{feat_key}.npy')
        X_glob = np.load(f'results/features/X_glob_{feat_key}.npy')
        y = np.load(f'results/features/y_{feat_key}.npy')
    except FileNotFoundError:
        print("Features not found. Please run --stage extract first.")
        return

    # The added columns come after the original ones, so ablating them is just slicing.
    if args.no_lens:
        X_seq = X_seq[:, :, :SEQ_BASE_DIM]
    if args.no_answer_ll:
        X_glob = X_glob[:, :GLOB_BASE_DIM]
    print(f"Input features: seq {X_seq.shape[1:]}, global {X_glob.shape[1:]}")

    X_seq_train, X_seq_test, X_glob_train, X_glob_test, y_train, y_test = train_test_split(
        X_seq, X_glob, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )

    N, L, F_seq = X_seq_train.shape
    scaler_seq = RobustScaler()
    X_seq_train = scaler_seq.fit_transform(X_seq_train.reshape(-1, F_seq)).reshape(N, L, F_seq)
    X_seq_test = scaler_seq.transform(X_seq_test.reshape(-1, F_seq)).reshape(X_seq_test.shape[0], L, F_seq)

    scaler_glob = RobustScaler()
    X_glob_train = scaler_glob.fit_transform(X_glob_train)
    X_glob_test = scaler_glob.transform(X_glob_test)

    skf = StratifiedKFold(n_splits=config.n_inner_folds, shuffle=True, random_state=config.seed)
    feat_dim = config.hidden_dim + (config.hidden_dim // 2)
    oof_features = np.zeros((len(y_train), feat_dim))
    test_features_accum = np.zeros((len(y_test), feat_dim))
    oof_logits = np.zeros(len(y_train))
    test_logits = np.zeros((config.n_inner_folds, len(y_test)))

    jobs = [(fold, tr_idx, val_idx, X_seq_train, X_glob_train, y_train, X_seq_test, X_glob_test, config)
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_seq_train, y_train))]
    if config.fold_workers > 1:
        # The model is small, so the folds share one GPU; "spawn" is required with CUDA.
        with ProcessPoolExecutor(config.fold_workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_train_fold, *zip(*jobs)))
    else:
        results = [_train_fold(*job) for job in jobs]

    for fold, (val_idx, feats_val, feats_test, logit_val, logit_test, best_auc) in enumerate(results):
        print(f"Fold {fold+1}/{config.n_inner_folds}: best validation AUC {best_auc:.4f}")
        oof_features[val_idx] = feats_val
        test_features_accum += feats_test
        oof_logits[val_idx] = logit_val
        test_logits[fold] = logit_test

    # Deep-network-only reference, useful when comparing ablations. The OOF number is slightly
    # optimistic because each fold's best epoch was picked on that same validation fold.
    print(f"Deep-only AUC: OOF logit (train) {roc_auc_score(y_train, oof_logits):.4f} | "
          f"fold-averaged logit (test) {roc_auc_score(y_test, test_logits.mean(axis=0)):.4f}")

    key = _key(args)
    os.makedirs('results/oof', exist_ok=True)
    # "vector" stack mode: 576-d features (test = average over the 5 unrelated feature spaces)
    np.save(f'results/oof/X_train_deep_{key}.npy', oof_features)
    np.save(f'results/oof/X_test_deep_{key}.npy', test_features_accum / config.n_inner_folds)
    # "logit" stack mode: one logit per fold network + the shared-space global features
    np.save(f'results/oof/oof_logits_{key}.npy', oof_logits)
    np.save(f'results/oof/test_logits_{key}.npy', test_logits)
    np.save(f'results/oof/X_glob_train_{key}.npy', X_glob_train)
    np.save(f'results/oof/X_glob_test_{key}.npy', X_glob_test)
    np.save(f'results/oof/y_train_{key}.npy', y_train)
    np.save(f'results/oof/y_test_{key}.npy', y_test)
    print("Stage 2 & 3 completed. OOF outputs saved.")


def stage_4_ensemble(args, config):
    print(f"\n--- Stage 4: Log-Odds Ensemble Meta-Learning (stack mode: {args.stack_mode}) ---")
    key = _key(args)
    try:
        y_train = np.load(f'results/oof/y_train_{key}.npy')
        y_test = np.load(f'results/oof/y_test_{key}.npy')
        if args.stack_mode == "vector":
            X_train = np.load(f'results/oof/X_train_deep_{key}.npy')
            X_tests = [np.load(f'results/oof/X_test_deep_{key}.npy')]
        else:
            oof_logits = np.load(f'results/oof/oof_logits_{key}.npy')
            test_logits = np.load(f'results/oof/test_logits_{key}.npy')
            glob_train = np.load(f'results/oof/X_glob_train_{key}.npy')
            glob_test = np.load(f'results/oof/X_glob_test_{key}.npy')
            X_train = np.column_stack([oof_logits, glob_train])
            # One test matrix per fold network: the stacker is trained on rows that each carry a
            # single network's logit, so it is also applied that way; the outputs are averaged below.
            X_tests = [np.column_stack([test_logits[k], glob_test]) for k in range(len(test_logits))]
    except FileNotFoundError:
        print("OOF outputs not found. Please run --stage train_oof first.")
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_tests = [scaler.transform(x) for x in X_tests]

    ensemble = get_ensemble(config.seed)
    calibrated_model, _ = calibrate_ensemble(ensemble, X_train, y_train)

    probs = np.mean([calibrated_model.predict_proba(x)[:, 1] for x in X_tests], axis=0)

    thresholds = find_best_thresholds(probs, y_test)
    results = evaluate_all(probs, y_test, thresholds['youden'])
    uncertainty = compute_uncertainty_metrics(probs, y_test, (probs >= thresholds['youden']).astype(int))

    print("\nResults Summary:")
    print(f"AUC: {results['auc']:.4f}, F1: {results['f1']:.4f}, Acc: {results['acc']:.4f}")

    os.makedirs('results/plots', exist_ok=True)
    plot_path = f"results/plots/results_{key}.png"
    plot_publication_results(
        probs, y_test, (probs >= thresholds['youden']).astype(int),
        uncertainty, results['auc'], results['ap'], results['cm'],
        plot_path
    )
    print(f"Saved results to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="MultiHaluDet 4-Stage Execution Pipeline")
    parser.add_argument("--dataset", type=str, default="halueval", choices=["halueval", "triviaqa"])
    parser.add_argument("--lang", type=str, default="en", choices=["en", "fr", "bn", "am", "vi"], help="Language variant of the dataset")
    parser.add_argument("--model", type=str, default="mistral-7b", choices=PIPELINE_MODELS)
    parser.add_argument("--stage", type=str, default="all", choices=["all", "extract", "train_oof", "ensemble"],
                        help="all: Run full pipeline; extract: Feature Extraction; train_oof: Multi-Scale Modeling & OOF; ensemble: Ensemble Meta-Learning")
    parser.add_argument("--extract_batch_size", type=int, default=4,
                        help="Batch size for LLM feature extraction; lower it (e.g. 2) if you hit CUDA OOM")

    # Stacking and ablation options
    parser.add_argument("--stack_mode", type=str, default="logit", choices=["logit", "vector"],
                        help="logit: stack fold-network logits + global features; vector: stack the 576-d features (original)")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed (changes the split and the folds)")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for OOF/plot files so several runs (seeds, ablations) do not overwrite each other")
    parser.add_argument("--no_mixup", action="store_true", help="Ablation: disable Mixup")
    parser.add_argument("--no_cutmix", action="store_true", help="Ablation: disable CutMix")
    parser.add_argument("--no_layer_delta", action="store_true", help="Ablation: disable layer-to-layer delta channels")
    parser.add_argument("--no_layer_scale", action="store_true", help="Ablation: disable the xL scaling of layer weights")
    parser.add_argument("--rank_weight", type=float, default=None,
                        help="Override the pairwise AUC loss weight (0 disables it)")
    parser.add_argument("--no_lens", action="store_true",
                        help="Ablation: drop the 2 logit-lens channels per layer (seq 14 -> 12)")
    parser.add_argument("--no_answer_ll", action="store_true",
                        help="Ablation: drop the 4 answer log-likelihood features (global 34 -> 30)")
    args = parser.parse_args()

    config = get_config()
    if args.seed is not None:
        config.seed = args.seed
    if args.no_mixup:
        config.use_mixup = False
    if args.no_cutmix:
        config.use_cutmix = False
    if args.no_layer_delta:
        config.use_layer_delta = False
    if args.no_layer_scale:
        config.scale_layer_weights = False
    if args.rank_weight is not None:
        config.rank_loss_weight = args.rank_weight

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.stage in ["extract", "all"]:
        stage_1_extract(args, config)
    if args.stage in ["train_oof", "all"]:
        stage_2_3_train_oof(args, config)
    if args.stage in ["ensemble", "all"]:
        stage_4_ensemble(args, config)

if __name__ == "__main__":
    main()
