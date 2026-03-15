# =============================================================================
# config.py — single source of truth for all knobs
# =============================================================================

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-0.6B"
OUTPUT_DIR    = "./checkpoints"
HF_REPO_ID    = None   # set to "username/repo" to push after training

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_RANK         = 64
LORA_ALPHA        = 128          # 2x rank
LORA_TARGET       = ["gate_proj", "up_proj", "down_proj"]   # MLP only, frozen attn
LORA_LAYERS_FRAC  = 0.5          # only tune top 50% of layers; 0.0 = all layers

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH  = "./data/train.jsonl"
MAX_EXAMPLES  = 14              # None = full dataset; set small for smoke tests

# ── Generation ────────────────────────────────────────────────────────────────
BATCH_SIZE            = 10       # problems per gradient accumulation window
NUM_GENERATIONS       = 4       # completions per problem (G in GRPO)
MAX_SEQ_LEN           = 32768    # total context window
MAX_COMPLETION_TOKENS = 2000    # max new tokens per completion
TEMPERATURE           = 0.7
REASONING_EFFORT      = "low" # gpt-oss / Qwen3 thinking budget

# ── Memory ────────────────────────────────────────────────────────────────────
SPARSE_LOGIT_THRESH = 1e-3       # drop tokens with prob < 0.01%; keeps ~hundreds per pos
ENTROPY_CHUNK       = 512       # vocab chunk size for entropy computation
LOGPROB_CHUNK       = 512       # vocab chunk size for logprob computation
SGLANG_MEM_FRAC     = 0.2       # fraction of GPU VRAM for SGLang KV cache + weights

# ── Training ──────────────────────────────────────────────────────────────────
LR              = 1e-4
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.05
LR_SCHEDULER    = "cosine"
OPTIMIZER       = "adamw_8bit"   # bitsandbytes 8-bit adam; falls back to adamw if unavailable
GRAD_ACCUM_STEPS = 25            # == BATCH_SIZE; one optimizer step per batch
KL_COEFF        = 0.04           # KL penalty weight; 0.0 disables KL term
CLIP_RATIO      = 0.2            # PPO-style clip on policy ratio (inactive in pure online mode)

# ── Reward shaping ────────────────────────────────────────────────────────────
REWARD_COMPILE      = 1.0        # flat bonus for compiling successfully
REWARD_PER_TEST     = 1.0        # per passing test case
REWARD_LENGTH_PENALTY = 0.01     # per token over MIN_COMPLETION_TOKENS
MIN_COMPLETION_TOKENS = 64       # below this: mild penalty to discourage degenerate shorts
REWARD_ERROR_ENGAGE  = 0.1       # Pass 2 only: bonus if thinking mentions compiler error tokens

# ── Verification ──────────────────────────────────────────────────────────────
VERIFY_TIMEOUT_S  = 10           # per-test-case wall time (seconds)
VERIFY_WORKERS    = 32           # parallel gcc worker processes

# ── Checkpointing ─────────────────────────────────────────────────────────────
SAVE_STEPS        = 50           # save LoRA adapter every N optimizer steps
LOG_STEPS         = 1
