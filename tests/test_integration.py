"""
Integration Tests for TAC-LAnoBERT Phase 3

Tests the complete pipeline:
1. Preprocess with timestamp extraction
2. Dataset loading with Time2Vec
3. Model forward pass
4. Training step
5. Inference with Memory Queue
"""

import os
import sys
import tempfile
import shutil
import torch
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertTokenizer
from tac_lanobert.dataset_tac import TACLogLineDataset
from tac_lanobert.model import TACLAnoBERT, TACConfig
from tac_lanobert.time_delta import TimestampExtractor
from transformers import BertConfig


class TestIntegration:
    """Integration tests for TAC-LAnoBERT."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    @pytest.fixture
    def tokenizer(self, temp_dir):
        """Create a test tokenizer."""
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
        tok.save_pretrained(temp_dir)
        return tok
    
    @pytest.fixture
    def sample_logs(self, temp_dir):
        """Create sample log file with BGL format."""
        log_path = os.path.join(temp_dir, "test.log")
        # .timestamps file must be named after the log file (replace extension)
        timestamp_path = log_path.replace(".log", ".timestamps")
        
        # Create log lines with BGL format (Unix timestamp at start)
        logs = [
            "1117838570 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
            "1117838571 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO data cache parity error corrected",
            "1117838580 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL data TLB error interrupt",
            "1117838590 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO timer interrupt",
            "1117838600 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO timer interrupt",
        ]
        
        with open(log_path, 'w') as f:
            f.write('\n'.join(logs))
        
        # Extract timestamps and write as floats (one per line) — NOT raw log lines
        extractor = TimestampExtractor('bgl')
        with open(timestamp_path, 'w') as f:
            for line in logs:
                ts = extractor.extract_timestamp(line)
                f.write(f"{ts if ts is not None else 0.0}\n")
        
        return log_path, timestamp_path
    
    def test_timestamp_extraction(self, sample_logs):
        """Test timestamp extraction from BGL logs."""
        log_path, timestamp_path = sample_logs
        
        # Read extracted timestamps
        with open(timestamp_path, 'r') as f:
            timestamps = [float(line.strip()) for line in f]
        
        assert len(timestamps) == 5
        assert timestamps[0] == 1117838570
        assert timestamps[-1] == 1117838600
        
        # Check monotonic increase
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], "Timestamps should be monotonic"
    
    def test_dataset_with_time2vec(self, tokenizer, sample_logs, temp_dir):
        """Test TACLogLineDataset with Time2Vec enabled."""
        log_path, _ = sample_logs
        
        dataset = TACLogLineDataset(
            tokenizer=tokenizer,
            file_path=log_path,
            max_len=128,
            log_format='bgl'
        )
        
        # Check dataset loaded correctly
        assert len(dataset) == 5
        assert dataset.delta_t is not None
        assert len(dataset.delta_t) == 5
        
        # Check first delta_t is 0 (first event)
        assert dataset.delta_t[0] == 0.0, "First delta_t should be 0"
        
        # Check subsequent delta_t values are positive
        for i in range(1, len(dataset.delta_t)):
            assert dataset.delta_t[i] >= 0.0, f"delta_t[{i}] should be non-negative"
        
        # Check item contains delta_t field
        item = dataset[0]
        assert 'delta_t' in item
        assert isinstance(item['delta_t'], list)
    
    def test_tac_model_baseline(self):
        """Test TAC model in baseline mode (no TAC features)."""
        bert_config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=128
        )
        
        tac_config = TACConfig(mode='baseline')
        model = TACLAnoBERT(bert_config, tac_config)
        
        # Check features disabled
        assert model.time2vec is None
        assert model.memory_queue is None
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert outputs.logits.shape == (2, 32, 1000)
    
    def test_tac_model_time_only(self):
        """Test TAC model with Time2Vec only."""
        bert_config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=128
        )
        
        tac_config = TACConfig(mode='time_only', num_periodic=5)
        model = TACLAnoBERT(bert_config, tac_config)
        
        # Check Time2Vec enabled, Memory disabled
        assert model.time2vec is not None
        assert model.memory_queue is None
        
        # Forward pass with delta_t
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        delta_t = torch.rand(batch_size, seq_len) * 10  # Normalized deltas
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            delta_t=delta_t
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, 1000)
    
    def test_tac_model_memory_only(self):
        """Test TAC model with Memory Queue only."""
        bert_config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=128
        )
        
        tac_config = TACConfig(mode='memory_only', queue_capacity=10, min_samples=2)
        model = TACLAnoBERT(bert_config, tac_config)
        
        # Check Memory enabled, Time2Vec disabled
        assert model.time2vec is None
        assert model.memory_queue is not None
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert outputs.logits.shape == (2, 32, 1000)
        
        # Test [CLS] extraction and memory queue
        cls_vector = model.get_cls_vector(input_ids, attention_mask)
        assert cls_vector.shape == (2, 128)
        
        # Push to memory queue
        model.memory_queue.push(cls_vector[0])
        assert len(model.memory_queue) == 1
        
        # Push more vectors
        for _ in range(5):
            model.memory_queue.push(cls_vector[0])
        
        assert len(model.memory_queue) == 6
        
        # Test Mahalanobis distance (should not error even with few samples)
        dist = model.memory_queue.mahalanobis_distance(cls_vector[0])
        assert dist >= 0.0
    
    def test_tac_model_full(self):
        """Test TAC model with all features enabled."""
        bert_config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=128
        )
        
        tac_config = TACConfig(
            mode='full',
            num_periodic=5,
            queue_capacity=10,
            min_samples=2
        )
        model = TACLAnoBERT(bert_config, tac_config)
        
        # Check both features enabled
        assert model.time2vec is not None
        assert model.memory_queue is not None
        
        # Forward pass with delta_t
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        delta_t = torch.rand(batch_size, seq_len) * 10
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            delta_t=delta_t,
            output_hidden_states=True
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, 1000)
        
        # Test [CLS] extraction with delta_t
        cls_vector = model.get_cls_vector(input_ids, attention_mask, delta_t)
        assert cls_vector.shape == (batch_size, 128)
    
    def test_training_step(self):
        """Test a single training step."""
        bert_config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=128
        )
        
        tac_config = TACConfig(mode='full', num_periodic=5, queue_capacity=10)
        model = TACLAnoBERT(bert_config, tac_config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch with labels
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        delta_t = torch.rand(batch_size, seq_len) * 10
        labels = input_ids.clone()
        
        # Forward pass
        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            delta_t=delta_t,
            labels=labels
        )
        
        # Check loss
        assert outputs.loss is not None
        assert outputs.loss.item() > 0
        
        # Backward pass
        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        
        # Check gradients flowed
        assert model.bert.bert.embeddings.word_embeddings.weight.grad is not None
        if model.time2vec is not None:
            assert model.time2vec.omega_linear.grad is not None
    
    def test_save_and_load(self, temp_dir):
        """Test model save and load."""
        bert_config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=128
        )
        
        tac_config = TACConfig(mode='full', num_periodic=5)
        model = TACLAnoBERT(bert_config, tac_config)
        
        # Save
        save_dir = os.path.join(temp_dir, "model")
        model.save_pretrained(save_dir)
        
        # Check files exist
        assert os.path.exists(os.path.join(save_dir, "tac_config.json"))
        assert os.path.exists(os.path.join(save_dir, "time2vec.pt"))
        assert os.path.exists(os.path.join(save_dir, "config.json"))
        
        # Load
        loaded_model = TACLAnoBERT.from_pretrained(save_dir)
        
        # Check config matches
        assert loaded_model.tac_config.mode == 'full'
        assert loaded_model.tac_config.num_periodic == 5
        assert loaded_model.time2vec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
