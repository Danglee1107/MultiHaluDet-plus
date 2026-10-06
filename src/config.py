import torch
from dotenv import load_dotenv

# Loads HF_TOKEN from .env into the environment; huggingface_hub reads it automatically.
load_dotenv()

class Config:
    seed = 42
    n_inner_folds = 5
    test_size = 0.20
    
    hidden_dim = 384
    num_heads = 8
    num_layers = 6
    scales = [1, 2, 4, 8]
    mlp_hidden_dim = 288
    dropout = 0.25
    
    # Speed settings (OPTIMIZER.md S7 R3). The paper's values are 28, 2e-4 and ema_decay 0.999;
    # restore all three together to reproduce the paper's training.
    batch_size = 128
    epochs = 45
    learning_rate = 4e-4
    weight_decay = 6e-5
    patience = 15
    min_lr = 1e-7
    warmup_epochs = 5
    
    use_mixup = True
    mixup_alpha = 0.15
    use_cutmix = True
    cutmix_alpha = 0.15
    label_smoothing = 0.02
    contrastive_weight = 0.20
    contrastive_temp = 0.04
    grad_clip = 0.5
    
    use_ema = True
    ema_decay = 0.995   # same averaging window in samples as 0.999 at batch 28
    use_amp = True      # bfloat16 forward pass on GPU; no effect on CPU
    fold_workers = 3    # inner folds trained at the same time; 1 = one after another
    use_swa = True
    swa_start = 30
    swa_lr = 5e-5
    
    # Group A improvements. Each can be switched off from run_pipeline.py for ablation.
    use_layer_delta = True       # add x_l - x_(l-1) channels to the sequential input (12 -> 24)
    scale_layer_weights = True   # softmax layer weights times L, so initialisation is identity
    rank_loss_weight = 0.1       # weight of the pairwise AUC loss; 0 disables it
    rank_loss_tau = 1.0          # temperature of the pairwise AUC loss

    use_class_weights = False
    minority_oversample = 1.0
    decision_bias = 0.0
    
    focal_alpha_pos = 0.5
    focal_alpha_neg = 0.5
    focal_gamma = 2.0

    n_sample_layers  = 32
    anchor_fractions = [0.25, 0.50, 0.75, 1.00]
    # float16 needs a GPU; the CPU path exists for the test/ smoke run.
    model_dtype      = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_REGISTRY = {
    "mistral-7b"   : "mistralai/Mistral-7B-Instruct-v0.2",
    "llama2-7b"    : "meta-llama/Llama-2-7b-hf",
    "qwen2.5-0.5b" : "Qwen/Qwen2.5-0.5B",
}

# Small models for test/ only; run_pipeline.py does not offer them on the command line.
TEST_ONLY_MODELS = {"qwen2.5-0.5b"}
PIPELINE_MODELS = [name for name in MODEL_REGISTRY if name not in TEST_ONLY_MODELS]

def get_config():
    return Config()
