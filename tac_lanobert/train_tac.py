"""
TAC-LAnoBERT Training Script: Train with Time2Vec + Memory Queue support.

This extends lanobert.train with TAC-specific features:
- Time2Vec embedding injection
- Delta_t data loading
- Feature-flagged training modes
"""

import argparse
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tac_lanobert.dataset_tac import TACLogLineDataset
from tac_lanobert.tokenizer_tac import load_tokenizer, vocab_path_for
from tac_lanobert.utils_tac import ensure_dir, load_config, set_seed

from .model import TACLAnoBERT, TACConfig


class TACDataCollator:
    """
    Custom data collator that handles `delta_t` field in addition to MLM fields.

    DataCollatorForLanguageModeling only handles input_ids/attention_mask/labels.
    When Time2Vec is enabled, dataset items also contain `delta_t` (list of floats
    per token). This collator:
    1. Extracts `delta_t` from each item before passing to the base collator
    2. Stacks and pads `delta_t` tensors to match the padded sequence length
    3. Returns the combined batch dict with both MLM fields and `delta_t`
    """

    def __init__(self, base_collator, pad_token_id: int = 0):
        self.base_collator = base_collator
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate delta_t from the rest
        delta_t_list = None
        if features and "delta_t" in features[0]:
            delta_t_list = [f.pop("delta_t") for f in features]

        # Standard MLM collation (handles padding of input_ids, labels, attention_mask)
        batch = self.base_collator(features)

        # Re-attach and pad delta_t
        if delta_t_list is not None:
            # Get the padded length from batch
            padded_len = batch["input_ids"].shape[1]
            padded_delta_t = torch.zeros(
                len(delta_t_list), padded_len, dtype=torch.float32
            )
            for i, dt in enumerate(delta_t_list):
                dt_tensor = torch.tensor(dt, dtype=torch.float32)
                # Truncate or pad to match padded_len
                actual_len = min(len(dt_tensor), padded_len)
                padded_delta_t[i, :actual_len] = dt_tensor[:actual_len]
            batch["delta_t"] = padded_delta_t

        return batch


def build_tac_model(vocab_size: int, max_len: int, tac_config: TACConfig, attn_implementation: str = "sdpa"):
    """
    Create a fresh TAC-LAnoBERT model.
    
    Args:
        vocab_size: Vocabulary size
        max_len: Max sequence length
        tac_config: TAC feature configuration
        attn_implementation: Attention implementation ('sdpa' or 'eager')
    
    Returns:
        TACLAnoBERT model
    """
    bert_config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # BERT-base defaults
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    
    model = TACLAnoBERT(bert_config, tac_config)
    
    print(f"[train_tac] Created TAC-LAnoBERT:")
    print(model.get_config_summary())
    print(f"[train_tac] Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_tac(cfg, vocab_file: Optional[str] = None) -> str:
    """
    Train TAC-LAnoBERT with MLM objective.
    
    Args:
        cfg: Config object loaded from YAML
        vocab_file: Optional path to vocabulary file
    
    Returns:
        Path to saved model directory
    """
    tcfg = cfg.get("train", {})
    set_seed(int(tcfg.get("seed", 42)))

    vocab_file = vocab_file or vocab_path_for(cfg)
    max_len = int(tcfg.get("max_len", 512))

    # Enable TF32 for speedup
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Detect MPS (Apple Silicon)
    _has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # Load TAC config from YAML
    tac_dict = cfg.get("tac", {})
    if not tac_dict or not tac_dict.get("enabled", False):
        print("[train_tac] WARNING: TAC features not enabled in config!")
        print("[train_tac] Set 'tac.enabled: true' in config to use TAC features")
        tac_config = TACConfig(mode="baseline")
    else:
        tac_config = TACConfig(
            mode=tac_dict.get("mode", "full"),
            enable_time2vec=tac_dict.get("time2vec", {}).get("enabled", True),
            num_periodic=tac_dict.get("time2vec", {}).get("num_periodic", 15),
            enable_memory=tac_dict.get("memory", {}).get("enabled", True),
            queue_capacity=tac_dict.get("memory", {}).get("queue_capacity", 128),
            min_samples=tac_dict.get("memory", {}).get("min_samples", 10),
            scoring_alpha=tac_dict.get("scoring", {}).get("alpha", 0.5),
        )

    # Load tokenizer
    tokenizer = load_tokenizer(vocab_file, max_len=max_len)
    print(f"[train_tac] loaded tokenizer (vocab={tokenizer.vocab_size})")

    # Build model
    attn = str(tcfg.get("attn_implementation", "sdpa"))
    model = build_tac_model(
        vocab_size=tokenizer.vocab_size,
        max_len=max_len,
        tac_config=tac_config,
        attn_implementation=attn
    )

    # Load dataset with Time2Vec support
    use_time2vec = tac_config.enable_time2vec
    log_format = cfg.get("dataset", "bgl").lower()
    
    print(f"[train_tac] loading dataset (Time2Vec={'enabled' if use_time2vec else 'disabled'})")
    
    if use_time2vec:
        full_dataset = TACLogLineDataset(
            tokenizer=tokenizer,
            file_path=cfg.get_path("paths.train_normal"),
            max_len=max_len,
            log_format=log_format,
            load_timestamps=True,
        )
    else:
        full_dataset = TACLogLineDataset(
            tokenizer=tokenizer,
            file_path=cfg.get_path("paths.train_normal"),
            max_len=max_len,
            log_format=log_format,
            load_timestamps=False,
        )

    # Train/eval split
    eval_ratio = float(tcfg.get("eval_ratio", 0.01))
    eval_size = max(1, int(len(full_dataset) * eval_ratio))
    train_size = len(full_dataset) - eval_size
    
    from torch.utils.data import random_split
    train_dataset, eval_dataset = random_split(
        full_dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(int(tcfg.get("seed", 42))),
    )
    print(f"[train_tac] examples: train={train_size:,} eval={eval_size:,}")

    # Data collator for MLM (with TAC wrapper to handle delta_t)
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(tcfg.get("mlm_probability", 0.15)),
        pad_to_multiple_of=8,
    )
    collator = TACDataCollator(base_collator, pad_token_id=tokenizer.pad_token_id or 0)

    model_dir = ensure_dir(cfg.get_path("paths.model_dir"))
    eval_steps = int(tcfg.get("eval_steps", tcfg.get("save_steps", 50000)))
    seed = int(tcfg.get("seed", 42))

    # Compute warmup steps
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.1))
    num_epochs = float(tcfg.get("num_train_epochs", 10))
    batch_size = int(tcfg.get("per_device_train_batch_size", 8))
    total_steps = int(len(train_dataset) / max(batch_size, 1) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=model_dir,
        seed=seed,
        data_seed=seed,
        full_determinism=bool(tcfg.get("full_determinism", False)),
        num_train_epochs=num_epochs,
        max_steps=int(tcfg.get("max_steps", -1)),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 64)),
        learning_rate=float(tcfg.get("learning_rate", 5e-5)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        adam_beta2=float(tcfg.get("adam_beta2", 0.98)),
        adam_epsilon=float(tcfg.get("adam_epsilon", 1e-6)),
        bf16=bool(tcfg.get("bf16", torch.cuda.is_available() and not _has_mps)),
        fp16=bool(tcfg.get("fp16", _has_mps)),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(tcfg.get("logging_steps", 1000)),
        dataloader_num_workers=0 if _has_mps else 4,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    resume = tcfg.get("resume_from_checkpoint", None)
    if resume is False:
        resume = None
    elif resume is None or resume is True:
        # Auto-detect latest checkpoint to resume training
        import glob
        checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
        if len(checkpoints) > 0:
            resume = True
            print(f"[train_tac] Found {len(checkpoints)} checkpoints! Will resume from the latest.")
        else:
            resume = None
            print("[train_tac] No checkpoints found. Starting from scratch.")
    
    print("[train_tac] start" + (f" (resuming from checkpoint)" if resume else ""))
    trainer.train(resume_from_checkpoint=resume)

    final_dir = os.path.join(model_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save TAC config copy
    import shutil
    config_path = cfg._source_path if hasattr(cfg, '_source_path') else None
    if config_path and os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(model_dir, "config_used.yaml"))
    
    print(f"[train_tac] saved final model -> {final_dir}")
    return final_dir


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="TAC-LAnoBERT MLM training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--vocab_file", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    train_tac(cfg, vocab_file=args.vocab_file)


if __name__ == "__main__":
    main()
