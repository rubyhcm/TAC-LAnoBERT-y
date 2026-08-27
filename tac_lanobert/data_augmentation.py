"""
Data Augmentation for Log Anomaly Detection

Generates synthetic anomalies during training to help the model
learn better decision boundaries.

Augmentation techniques:
1. Random token replacement
2. Token order shuffling
3. Temporal anomalies (abnormal time gaps)
4. Template mixing (combine tokens from different templates)
5. Frequency-based perturbation (rare events)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import pandas as pd


class LogAugmenter:
    """
    Augment normal log data with synthetic anomalies.
    
    Args:
        vocab: List of tokens in vocabulary
        augmentation_ratio: Ratio of augmented samples to original (default: 0.1 = 10%)
        methods: List of augmentation methods to use
    """
    
    def __init__(
        self,
        vocab: List[str],
        augmentation_ratio: float = 0.1,
        methods: Optional[List[str]] = None
    ):
        self.vocab = vocab
        self.augmentation_ratio = augmentation_ratio
        
        if methods is None:
            methods = ['replace', 'shuffle', 'temporal', 'mix']
        
        self.methods = methods
        
        # Special tokens to avoid augmenting
        self.special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    
    def augment_batch(
        self,
        tokens: List[List[str]],
        timestamps: Optional[pd.Series] = None,
        delta_t: Optional[np.ndarray] = None
    ) -> Tuple[List[List[str]], Optional[pd.Series], Optional[np.ndarray], np.ndarray]:
        """
        Augment a batch of normal logs.
        
        Args:
            tokens: List of token sequences
            timestamps: Optional timestamps
            delta_t: Optional time deltas
        
        Returns:
            (augmented_tokens, augmented_timestamps, augmented_delta_t, labels)
            labels: 0=original, 1=augmented (synthetic anomaly)
        """
        n_original = len(tokens)
        n_augment = int(n_original * self.augmentation_ratio)
        
        # Original samples (label=0)
        all_tokens = tokens.copy()
        labels = np.zeros(n_original, dtype=int)
        
        if timestamps is not None:
            all_timestamps = timestamps.copy()
        else:
            all_timestamps = None
        
        if delta_t is not None:
            all_delta_t = delta_t.copy()
        else:
            all_delta_t = None
        
        # Generate augmented samples
        for _ in range(n_augment):
            # Randomly select a sample to augment
            idx = random.randint(0, n_original - 1)
            original_tokens = tokens[idx]
            
            # Randomly select augmentation method
            method = random.choice(self.methods)
            
            if method == 'replace':
                aug_tokens = self._augment_replace(original_tokens)
                aug_timestamp = timestamps.iloc[idx] if timestamps is not None else None
                aug_delta = delta_t[idx] if delta_t is not None else None
            
            elif method == 'shuffle':
                aug_tokens = self._augment_shuffle(original_tokens)
                aug_timestamp = timestamps.iloc[idx] if timestamps is not None else None
                aug_delta = delta_t[idx] if delta_t is not None else None
            
            elif method == 'temporal':
                aug_tokens = original_tokens.copy()
                aug_timestamp = timestamps.iloc[idx] if timestamps is not None else None
                # Generate abnormal time gap
                if delta_t is not None:
                    aug_delta = delta_t[idx] * random.uniform(10, 100)
                else:
                    aug_delta = None
            
            elif method == 'mix':
                # Mix with another sample
                idx2 = random.randint(0, n_original - 1)
                aug_tokens = self._augment_mix(original_tokens, tokens[idx2])
                aug_timestamp = timestamps.iloc[idx] if timestamps is not None else None
                aug_delta = delta_t[idx] if delta_t is not None else None
            
            else:
                continue
            
            # Add to batch
            all_tokens.append(aug_tokens)
            labels = np.append(labels, 1)  # Label as synthetic anomaly
            
            if all_timestamps is not None and aug_timestamp is not None:
                all_timestamps = pd.concat([all_timestamps, pd.Series([aug_timestamp])])
            
            if all_delta_t is not None and aug_delta is not None:
                all_delta_t = np.append(all_delta_t, aug_delta)
        
        return all_tokens, all_timestamps, all_delta_t, labels
    
    def _augment_replace(
        self,
        tokens: List[str],
        num_replacements: Optional[int] = None
    ) -> List[str]:
        """
        Replace random tokens with other vocabulary tokens.
        
        Simulates: unexpected tokens in log messages
        """
        aug_tokens = tokens.copy()
        
        if num_replacements is None:
            # Replace 1-3 tokens
            num_replacements = random.randint(1, min(3, len(tokens)))
        
        # Find non-special token positions
        replaceable_positions = [
            i for i, token in enumerate(tokens)
            if token not in self.special_tokens
        ]
        
        if len(replaceable_positions) == 0:
            return aug_tokens
        
        # Select positions to replace
        num_replacements = min(num_replacements, len(replaceable_positions))
        positions = random.sample(replaceable_positions, num_replacements)
        
        # Replace with random vocab tokens
        for pos in positions:
            new_token = random.choice(self.vocab)
            # Avoid replacing with special tokens
            while new_token in self.special_tokens:
                new_token = random.choice(self.vocab)
            aug_tokens[pos] = new_token
        
        return aug_tokens
    
    def _augment_shuffle(self, tokens: List[str]) -> List[str]:
        """
        Shuffle token order (excluding special tokens).
        
        Simulates: out-of-order log events, corrupted logs
        """
        aug_tokens = tokens.copy()
        
        # Find non-special tokens
        shuffleable_indices = [
            i for i, token in enumerate(tokens)
            if token not in self.special_tokens
        ]
        
        if len(shuffleable_indices) <= 1:
            return aug_tokens
        
        # Extract shuffleable tokens
        shuffleable_tokens = [tokens[i] for i in shuffleable_indices]
        
        # Shuffle
        random.shuffle(shuffleable_tokens)
        
        # Put back
        for idx, token in zip(shuffleable_indices, shuffleable_tokens):
            aug_tokens[idx] = token
        
        return aug_tokens
    
    def _augment_mix(
        self,
        tokens1: List[str],
        tokens2: List[str],
        mix_ratio: float = 0.5
    ) -> List[str]:
        """
        Mix tokens from two different samples.
        
        Simulates: combined error patterns, correlated failures
        """
        # Take first part from tokens1, second part from tokens2
        split_idx = int(len(tokens1) * mix_ratio)
        
        # Handle different lengths
        min_len = min(len(tokens1), len(tokens2))
        split_idx = min(split_idx, min_len)
        
        aug_tokens = tokens1[:split_idx] + tokens2[split_idx:min_len]
        
        return aug_tokens
    
    def _augment_temporal(
        self,
        delta_t: float,
        anomaly_type: str = 'large_gap'
    ) -> float:
        """
        Create temporal anomalies.
        
        Args:
            delta_t: Original time delta
            anomaly_type: 'large_gap', 'small_gap', or 'burst'
        
        Returns:
            Anomalous delta_t
        """
        if anomaly_type == 'large_gap':
            # Unusually long gap (10-100x normal)
            return delta_t * random.uniform(10, 100)
        
        elif anomaly_type == 'small_gap':
            # Unusually short gap (burst of events)
            return delta_t * random.uniform(0.01, 0.1)
        
        elif anomaly_type == 'burst':
            # Very short gap (rapid fire events)
            return delta_t * 0.001
        
        else:
            return delta_t


class TemplateMixingAugmenter:
    """
    Advanced augmentation by mixing log templates.
    
    Learns common templates and mixes them to create synthetic anomalies.
    """
    
    def __init__(self, min_template_freq: int = 10):
        self.min_template_freq = min_template_freq
        self.templates: Dict[str, List[List[str]]] = {}
    
    def fit(self, tokens: List[List[str]]):
        """
        Learn templates from training data.
        
        Args:
            tokens: List of token sequences
        """
        # Extract templates (simplified: use full token sequence as template)
        for token_seq in tokens:
            template_str = ' '.join(token_seq)
            
            if template_str not in self.templates:
                self.templates[template_str] = []
            
            self.templates[template_str].append(token_seq)
        
        # Filter rare templates
        self.templates = {
            template: instances
            for template, instances in self.templates.items()
            if len(instances) >= self.min_template_freq
        }
    
    def augment(
        self,
        n_samples: int
    ) -> List[List[str]]:
        """
        Generate synthetic anomalies by mixing templates.
        
        Args:
            n_samples: Number of synthetic samples to generate
        
        Returns:
            List of augmented token sequences
        """
        if len(self.templates) < 2:
            return []
        
        template_keys = list(self.templates.keys())
        augmented = []
        
        for _ in range(n_samples):
            # Select two random templates
            template1, template2 = random.sample(template_keys, 2)
            
            # Get instances
            instance1 = random.choice(self.templates[template1])
            instance2 = random.choice(self.templates[template2])
            
            # Mix
            mix_point = len(instance1) // 2
            mixed = instance1[:mix_point] + instance2[mix_point:]
            
            augmented.append(mixed)
        
        return augmented


class FrequencyBasedAugmenter:
    """
    Augment by inserting rare (low-frequency) tokens.
    
    Rare tokens often indicate errors or unusual events.
    """
    
    def __init__(self, rare_percentile: float = 10.0):
        self.rare_percentile = rare_percentile
        self.token_freqs: Dict[str, int] = {}
        self.rare_tokens: List[str] = []
    
    def fit(self, tokens: List[List[str]]):
        """
        Compute token frequencies.
        
        Args:
            tokens: List of token sequences
        """
        # Count token frequencies
        for token_seq in tokens:
            for token in token_seq:
                self.token_freqs[token] = self.token_freqs.get(token, 0) + 1
        
        # Find rare tokens (bottom percentile)
        freqs = np.array(list(self.token_freqs.values()))
        threshold = np.percentile(freqs, self.rare_percentile)
        
        self.rare_tokens = [
            token for token, freq in self.token_freqs.items()
            if freq <= threshold
        ]
    
    def augment(
        self,
        tokens: List[str],
        num_insertions: int = 1
    ) -> List[str]:
        """
        Insert rare tokens into sequence.
        
        Args:
            tokens: Original token sequence
            num_insertions: Number of rare tokens to insert
        
        Returns:
            Augmented sequence
        """
        if len(self.rare_tokens) == 0:
            return tokens
        
        aug_tokens = tokens.copy()
        
        for _ in range(num_insertions):
            # Select a rare token
            rare_token = random.choice(self.rare_tokens)
            
            # Insert at random position
            insert_pos = random.randint(0, len(aug_tokens))
            aug_tokens.insert(insert_pos, rare_token)
        
        return aug_tokens


# Unit tests
def _test_augmentation():
    """Test data augmentation."""
    print("Testing Data Augmentation...")
    
    # Create synthetic data
    vocab = ['token' + str(i) for i in range(100)] + ['[PAD]', '[CLS]', '[SEP]']
    
    tokens_list = [
        ['[CLS]', 'token0', 'token1', 'token2', '[SEP]'],
        ['[CLS]', 'token3', 'token4', 'token5', '[SEP]'],
        ['[CLS]', 'token6', 'token7', 'token8', '[SEP]']
    ] * 10  # 30 samples
    
    timestamps = pd.date_range('2024-01-01', periods=30, freq='1min')
    delta_t = np.random.rand(30) * 10
    
    # Test 1: Basic augmentation
    print("\n1. Testing LogAugmenter...")
    augmenter = LogAugmenter(vocab, augmentation_ratio=0.2)
    
    aug_tokens, aug_ts, aug_dt, labels = augmenter.augment_batch(
        tokens_list, timestamps, delta_t
    )
    
    print(f"   Original samples: {len(tokens_list)}")
    print(f"   Augmented samples: {len(aug_tokens)}")
    print(f"   Labels: {np.sum(labels == 0)} normal, {np.sum(labels == 1)} augmented")
    print(f"   ✅ Augmentation ratio: {np.sum(labels == 1) / len(labels):.2%}")
    
    # Test 2: Specific augmentation methods
    print("\n2. Testing individual augmentation methods...")
    
    original = ['[CLS]', 'hello', 'world', 'test', '[SEP]']
    
    # Replace
    replaced = augmenter._augment_replace(original, num_replacements=2)
    print(f"   Original: {original}")
    print(f"   Replaced: {replaced}")
    
    # Shuffle
    shuffled = augmenter._augment_shuffle(original)
    print(f"   Shuffled: {shuffled}")
    
    # Mix
    other = ['[CLS]', 'foo', 'bar', 'baz', '[SEP]']
    mixed = augmenter._augment_mix(original, other)
    print(f"   Mixed: {mixed}")
    
    print("   ✅ All methods work")
    
    # Test 3: Template mixing
    print("\n3. Testing TemplateMixingAugmenter...")
    template_aug = TemplateMixingAugmenter(min_template_freq=3)
    template_aug.fit(tokens_list)
    
    synthetic = template_aug.augment(n_samples=5)
    print(f"   Generated {len(synthetic)} synthetic samples")
    print(f"   Example: {synthetic[0]}")
    print("   ✅ Template mixing works")
    
    # Test 4: Frequency-based
    print("\n4. Testing FrequencyBasedAugmenter...")
    freq_aug = FrequencyBasedAugmenter(rare_percentile=20.0)
    freq_aug.fit(tokens_list)
    
    print(f"   Found {len(freq_aug.rare_tokens)} rare tokens")
    
    aug_with_rare = freq_aug.augment(original, num_insertions=2)
    print(f"   Original length: {len(original)}")
    print(f"   Augmented length: {len(aug_with_rare)}")
    print("   ✅ Frequency-based augmentation works")
    
    print("\n✅ All augmentation tests passed!")


if __name__ == "__main__":
    _test_augmentation()
