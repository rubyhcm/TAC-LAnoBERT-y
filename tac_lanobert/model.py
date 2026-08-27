"""
TAC-LAnoBERT Model Wrapper: Feature-flagged extension of LAnoBERT.

Supports multiple modes:
- baseline: Pure LAnoBERT (no TAC features)
- time_only: LAnoBERT + Time2Vec
- memory_only: LAnoBERT + Session Memory
- full: LAnoBERT + Time2Vec + Session Memory (Full TAC)
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .time2vec import Time2VecLayer
from .memory_queue import SessionMemoryQueue
from .scoring import HybridProactiveScorer


@dataclass
class TACConfig:
    """Configuration for TAC-LAnoBERT features.
    
    The `mode` parameter provides preset configurations:
    - 'baseline': Pure LAnoBERT (no TAC features) - disable_time2vec=True, enable_memory=False
    - 'time_only': Time2Vec only - enable_time2vec=True, enable_memory=False
    - 'memory_only': Memory Queue only - enable_time2vec=False, enable_memory=True
    - 'full': Full TAC (Time2Vec + Memory) - enable_time2vec=True, enable_memory=True
    
    Note: The `mode` parameter overrides individual enable_* flags in `_apply_mode_flags()`.
    If you need custom combinations, instantiate with desired individual flags and 
    don't call `_apply_mode_flags()`, or set flags after initialization.
    
    Example:
        >>> # Use preset mode
        >>> config = TACConfig(mode='full')  # Both features enabled
        >>> 
        >>> # Custom combination (advanced)
        >>> config = TACConfig(mode='full', enable_time2vec=True, enable_memory=False)
        >>> # After _apply_mode_flags(), both will be True (mode overrides)
    """
    
    # Mode selection
    mode: str = "full"  # baseline | time_only | memory_only | full
    
    # Time2Vec settings
    enable_time2vec: bool = True
    num_periodic: int = 15
    
    # Memory Queue settings
    enable_memory: bool = True
    queue_capacity: int = 128
    min_samples: int = 10
    shrinkage_alpha: Optional[float] = None  # None = auto Ledoit-Wolf
    
    # Hybrid Scoring settings
    scoring_alpha: float = 0.5  # MLM weight (0.0 = pure Mahal, 1.0 = pure MLM)
    normalize_scores: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TACConfig':
        """Create from dictionary (e.g., from YAML)."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode,
            'enable_time2vec': self.enable_time2vec,
            'num_periodic': self.num_periodic,
            'enable_memory': self.enable_memory,
            'queue_capacity': self.queue_capacity,
            'min_samples': self.min_samples,
            'shrinkage_alpha': self.shrinkage_alpha,
            'scoring_alpha': self.scoring_alpha,
            'normalize_scores': self.normalize_scores,
        }


class TACLAnoBERT(nn.Module):
    """
    TAC-LAnoBERT: Time-Aware Continual LAnoBERT.
    
    Feature-flagged wrapper around BertForMaskedLM with optional:
    - Time2Vec embedding for temporal dynamics
    - Session Memory Queue for continual context
    - Hybrid Proactive Scoring (MLM + Mahalanobis)
    
    Args:
        bert_config: HuggingFace BertConfig
        tac_config: TAC feature configuration
    """
    
    # Required by HF Trainer when load_best_model_at_end=True
    _keys_to_ignore_on_save = None
    
    def __init__(self, bert_config: BertConfig, tac_config: TACConfig):
        super().__init__()
        
        self.bert_config = bert_config
        self.tac_config = tac_config
        
        # Apply mode-based feature flags
        self._apply_mode_flags()
        
        # Core BERT encoder (for MLM)
        self.bert = BertForMaskedLM(bert_config)
        
        # Time2Vec layer (optional)
        self.time2vec = None
        if self.tac_config.enable_time2vec:
            self.time2vec = Time2VecLayer(
                hidden_size=bert_config.hidden_size,
                num_periodic=self.tac_config.num_periodic
            )
        
        # Session Memory Queue (inference-only, not trained)
        self.memory_queue = None
        if self.tac_config.enable_memory:
            self.memory_queue = SessionMemoryQueue(
                capacity=self.tac_config.queue_capacity,
                hidden_dim=bert_config.hidden_size,
                min_samples=self.tac_config.min_samples,
                shrinkage_alpha=self.tac_config.shrinkage_alpha
            )
        
        # Hybrid Scorer (inference-only)
        self.scorer = HybridProactiveScorer(
            alpha=self.tac_config.scoring_alpha,
            normalize=self.tac_config.normalize_scores
        )
    
    def _apply_mode_flags(self):
        """Apply mode-specific feature flags."""
        mode = self.tac_config.mode.lower()
        
        if mode == "baseline":
            self.tac_config.enable_time2vec = False
            self.tac_config.enable_memory = False
        elif mode == "time_only":
            self.tac_config.enable_time2vec = True
            self.tac_config.enable_memory = False
        elif mode == "memory_only":
            self.tac_config.enable_time2vec = False
            self.tac_config.enable_memory = True
        elif mode == "full":
            self.tac_config.enable_time2vec = True
            self.tac_config.enable_memory = True
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from "
                           f"['baseline', 'time_only', 'memory_only', 'full']")
    
    def _build_embeddings(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build combined input embeddings: Token + Positional + TokenType + Time2Vec (optional).

        Shared by forward() and get_cls_vector() to avoid code duplication.

        Args:
            input_ids: (batch, seq_len)
            token_type_ids: (batch, seq_len) — zeros if None
            delta_t: (batch, seq_len) normalized time deltas (for Time2Vec)

        Returns:
            combined_embeddings: (batch, seq_len, hidden_size) after LayerNorm + Dropout
        """
        embeddings = self.bert.bert.embeddings.word_embeddings(input_ids)

        position_ids = torch.arange(
            input_ids.size(1), dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.bert.bert.embeddings.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.bert.bert.embeddings.token_type_embeddings(token_type_ids)

        combined_embeddings = embeddings + position_embeddings + token_type_embeddings

        # Add Time2Vec embedding if enabled and delta_t is provided
        if self.tac_config.enable_time2vec and self.time2vec is not None and delta_t is not None:
            time_embeddings = self.time2vec(delta_t)
            combined_embeddings = combined_embeddings + time_embeddings

        combined_embeddings = self.bert.bert.embeddings.LayerNorm(combined_embeddings)
        combined_embeddings = self.bert.bert.embeddings.dropout(combined_embeddings)

        return combined_embeddings

    def state_dict(self, *args, **kwargs):
        """Override state_dict to strip tied weights.
        
        Transformers >= 4.38 saves using safetensors by default, which throws an
        error if multiple tensors share the same physical memory.
        Since we wrap BertForMaskedLM inside nn.Module, Trainer's safetensors logic 
        doesn't auto-detect our tied weights. We manually strip the duplicate here.
        When loading via load_state_dict(strict=False), PyTorch's tied parameters 
        will auto-populate from the word_embeddings.
        """
        state = super().state_dict(*args, **kwargs)
        if "bert.cls.predictions.decoder.weight" in state:
            del state["bert.cls.predictions.decoder.weight"]
        if "bert.cls.predictions.decoder.bias" in state:
            del state["bert.cls.predictions.decoder.bias"]
        return state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,  # Accept extra fields from Trainer batch (e.g. special_tokens_mask)
    ):
        """
        Forward pass with optional Time2Vec embedding.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            token_type_ids: (batch, seq_len) token type IDs
            delta_t: (batch, seq_len) normalized time deltas (for Time2Vec)
            labels: (batch, seq_len) MLM labels (for training)
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return ModelOutput dict
        
        Returns:
            MaskedLMOutput or dict with loss, logits, hidden_states
        """
        batch_size, seq_len = input_ids.shape

        # Build combined embeddings (shared helper)
        combined_embeddings = self._build_embeddings(input_ids, token_type_ids, delta_t)

        # Prepare extended attention mask for encoder
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=combined_embeddings.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(combined_embeddings.dtype).min
        else:
            extended_attention_mask = None
        
        # Forward through BERT encoder
        outputs = self.bert.bert.encoder(
            combined_embeddings,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        sequence_output = outputs[0] if not return_dict else outputs.last_hidden_state
        
        # MLM head
        prediction_scores = self.bert.cls(sequence_output)
        
        # Compute loss if labels provided (training)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.bert_config.vocab_size),
                labels.view(-1)
            )
        
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        from transformers.modeling_outputs import MaskedLMOutput
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
    
    def get_cls_vector(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract [CLS] vector for memory queue.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            delta_t: (batch, seq_len) for Time2Vec
        
        Returns:
            cls_vector: (batch, hidden_size)
        """
        with torch.no_grad():
            # Build combined embeddings (reuse shared helper)
            combined_embeddings = self._build_embeddings(input_ids, None, delta_t)

            # Prepare attention mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask[:, None, None, :]
                extended_attention_mask = extended_attention_mask.to(dtype=combined_embeddings.dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(combined_embeddings.dtype).min
            else:
                extended_attention_mask = None

            # Forward through encoder
            encoder_outputs = self.bert.bert.encoder(
                combined_embeddings,
                attention_mask=extended_attention_mask,
                output_hidden_states=False,
                return_dict=True
            )

            # Extract [CLS] token (first token) from last hidden state
            cls_vector = encoder_outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        return cls_vector
    
    def save_pretrained(self, save_directory: str):
        """Save model and TAC config."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save BERT model
        self.bert.save_pretrained(save_directory)
        
        # Save TAC config
        tac_config_path = os.path.join(save_directory, "tac_config.json")
        with open(tac_config_path, 'w') as f:
            json.dump(self.tac_config.to_dict(), f, indent=2)
        
        # Save Time2Vec weights if enabled
        if self.time2vec is not None:
            time2vec_path = os.path.join(save_directory, "time2vec.pt")
            torch.save(self.time2vec.state_dict(), time2vec_path)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        tac_config: Optional[TACConfig] = None
    ):
        """Load model with TAC components."""
        import os
        import json
        
        # Load BERT model
        bert_model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        
        # Load TAC config if available
        if tac_config is None:
            tac_config_path = os.path.join(pretrained_model_name_or_path, "tac_config.json")
            if os.path.exists(tac_config_path):
                with open(tac_config_path, 'r') as f:
                    tac_config = TACConfig.from_dict(json.load(f))
            else:
                # Default to baseline mode
                tac_config = TACConfig(mode="baseline")
        
        # Create TAC model
        model = cls(bert_model.config, tac_config)
        model.bert = bert_model
        
        # Load Time2Vec weights if available
        if model.time2vec is not None:
            time2vec_path = os.path.join(pretrained_model_name_or_path, "time2vec.pt")
            if os.path.exists(time2vec_path):
                model.time2vec.load_state_dict(torch.load(time2vec_path))
        
        return model
    
    def reset_memory(self):
        """Reset session memory queue (for new session/file)."""
        if self.memory_queue is not None:
            self.memory_queue.reset()
        if self.scorer is not None:
            self.scorer.reset_stats()
    
    def get_config_summary(self) -> str:
        """Get human-readable config summary."""
        lines = [
            "TAC-LAnoBERT Configuration:",
            f"  Mode: {self.tac_config.mode}",
            f"  Time2Vec: {'✓' if self.tac_config.enable_time2vec else '✗'}",
            f"  Memory Queue: {'✓' if self.tac_config.enable_memory else '✗'}",
        ]
        
        if self.tac_config.enable_time2vec:
            lines.append(f"    - Periodic components: {self.tac_config.num_periodic}")
        
        if self.tac_config.enable_memory:
            lines.append(f"    - Queue capacity: {self.tac_config.queue_capacity}")
            lines.append(f"    - Min samples: {self.tac_config.min_samples}")
        
        lines.append(f"  Hybrid α: {self.tac_config.scoring_alpha}")
        
        return "\n".join(lines)


# Unit test
def _test_tac_model():
    """Test TAC-LAnoBERT model in different modes."""
    from transformers import BertConfig
    
    # Small BERT config for testing
    bert_config = BertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=512
    )
    
    batch_size = 2
    seq_len = 32
    
    # Test different modes
    modes = ["baseline", "time_only", "memory_only", "full"]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing mode: {mode}")
        print(f"{'='*60}")
        
        tac_config = TACConfig(mode=mode, queue_capacity=10, min_samples=2)
        model = TACLAnoBERT(bert_config, tac_config)
        
        print(model.get_config_summary())
        
        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        delta_t = torch.rand(batch_size, seq_len) * 10  # Normalized deltas
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            delta_t=delta_t,
            output_hidden_states=True
        )
        
        print(f"\n  Forward pass successful:")
        print(f"    Logits shape: {outputs.logits.shape}")
        
        # Test [CLS] extraction
        cls_vector = model.get_cls_vector(input_ids, attention_mask, delta_t)
        print(f"    [CLS] shape: {cls_vector.shape}")
        
        # Test memory queue (if enabled)
        if model.memory_queue is not None:
            for i in range(5):
                model.memory_queue.push(cls_vector[0])
            
            mahal = model.memory_queue.mahalanobis_distance(cls_vector[0])
            print(f"    Memory queue size: {len(model.memory_queue)}")
            print(f"    Mahalanobis dist: {mahal:.4f}")
        
        print(f"\n  ✅ Mode '{mode}' passed!")
    
    print(f"\n{'='*60}")
    print("✅ All TAC-LAnoBERT modes tested successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    _test_tac_model()
