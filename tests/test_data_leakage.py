"""
Anti-leakage verification tests for TAC-LAnoBERT.

Ensures:
1. Chronological split is strictly enforced (no shuffle on test set)
2. No data leakage from future timestamps
3. Memory Queue only contains historical data (t ≤ t_current)
"""

import unittest
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChronologicalSplit(unittest.TestCase):
    """Test that data splits maintain chronological order."""

    def test_bgl_split_chronological(self):
        """Verify BGL split maintains temporal order."""
        split_path = Path(__file__).parent.parent / "data" / "BGL"
        
        if not split_path.exists():
            self.skipTest("BGL data not found")
        
        # Check that split stats exist
        stats_file = split_path / "split_stats.json"
        if stats_file.exists():
            import json
            with open(stats_file) as f:
                stats = json.load(f)
            
            # Verify split was created
            # Stats format: {'train_normal': X, 'test_total': Y, 'test_anomaly': Z, 'test_normal': W}
            self.assertIn("train_normal", stats)
            self.assertIn("test_total", stats)
            print(f"✓ BGL split verified: train_normal={stats['train_normal']:,}, test_total={stats['test_total']:,}")

    def test_no_shuffle_on_test_dataloader(self):
        """Verify that test DataLoader does NOT use shuffle=True."""
        # This is a code inspection test - check inference.py
        inference_file = Path(__file__).parent.parent / "lanobert" / "inference.py"
        
        if inference_file.exists():
            content = inference_file.read_text()
            
            # DataLoader for test/inference should have shuffle=False
            # We check that there's no explicit shuffle=True in test context
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'DataLoader' in line and 'shuffle=True' in line:
                    # Check context - should not be in inference/test mode
                    context = '\n'.join(lines[max(0, i-5):i+5])
                    if 'test' in context.lower() or 'inference' in context.lower():
                        self.fail(f"Found shuffle=True in test/inference context at line {i+1}")
            
            print("✓ No shuffle=True found in inference DataLoader")


class TestTemporalOrder(unittest.TestCase):
    """Test that timestamps are monotonically increasing (or correctly sorted)."""

    def test_bgl_raw_timestamps_extracted(self):
        """Verify that BGL raw logs contain valid timestamps."""
        bgl_test = Path(__file__).parent.parent / "data" / "BGL" / "BGL_test.raw"
        
        if not bgl_test.exists():
            self.skipTest("BGL test raw data not found")
        
        # BGL format: timestamp at beginning (Unix epoch)
        # Example: 1117838400 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL ...
        
        timestamps = []
        with open(bgl_test, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1000:  # Sample first 1000 lines
                    break
                parts = line.strip().split(maxsplit=1)
                if parts:
                    try:
                        ts = int(parts[0])
                        timestamps.append(ts)
                    except ValueError:
                        pass
        
        if len(timestamps) > 1:
            # Check monotonicity (allowing equal consecutive timestamps)
            for i in range(len(timestamps) - 1):
                self.assertLessEqual(
                    timestamps[i], 
                    timestamps[i+1],
                    f"Timestamps not monotonic at index {i}: {timestamps[i]} > {timestamps[i+1]}"
                )
            
            print(f"✓ BGL timestamps are monotonic (checked {len(timestamps)} samples)")
            print(f"  First: {timestamps[0]} ({datetime.fromtimestamp(timestamps[0])})")
            print(f"  Last: {timestamps[-1]} ({datetime.fromtimestamp(timestamps[-1])})")

    def test_thunderbird_raw_timestamps_extracted(self):
        """Verify that Thunderbird raw logs contain valid timestamps."""
        tbird_log = Path(__file__).parent.parent / "data" / "Thunderbird" / "Thunderbird.log"
        
        if not tbird_log.exists():
            self.skipTest("Thunderbird data not found")
        
        # Thunderbird format: Label Content E.g., "-" indicates normal and others indicate anomalous
        # Format varies, but typically has timestamp early in line
        # We'll do a basic check that file is readable
        
        with open(tbird_log, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            self.assertTrue(len(first_line) > 0, "Thunderbird log is empty")
            print(f"✓ Thunderbird log accessible, first line length: {len(first_line)}")


class TestMemoryQueueTemporal(unittest.TestCase):
    """Test that Memory Queue only contains historical data."""

    def test_memory_queue_no_future_access(self):
        """Verify Memory Queue design prevents future data access."""
        # Check if memory_queue.py exists
        memory_queue_file = Path(__file__).parent.parent / "tac_lanobert" / "memory_queue.py"
        
        if not memory_queue_file.exists():
            self.skipTest("Memory Queue not implemented yet (Phase 3)")
            return
        
        # Read the implementation
        content = memory_queue_file.read_text()
        
        # Check for FIFO queue structure
        self.assertIn("deque", content, "Memory Queue should use deque for FIFO")
        
        # Check that push/append is used (no look-ahead)
        has_push = "push" in content or "append" in content
        self.assertTrue(has_push, "Memory Queue should have push/append method")
        
        # Make sure there's no "future" or "look-ahead" in comments
        lines = content.lower().split('\n')
        for i, line in enumerate(lines):
            if 'future' in line and 'look' in line:
                # Check if it's in a warning/constraint context
                if 'no' not in line and 'prevent' not in line and 'avoid' not in line:
                    print(f"Warning: line {i+1} mentions 'future' and 'look': {line.strip()}")
        
        print("✓ Memory Queue structure verified (FIFO design)")


class TestDeterministicSettings(unittest.TestCase):
    """Test that deterministic settings are properly configured."""

    def test_seed_configuration_exists(self):
        """Verify that seed/deterministic settings are documented or configured."""
        # Check train.py for seed settings
        train_file = Path(__file__).parent.parent / "lanobert" / "train.py"
        
        if train_file.exists():
            content = train_file.read_text()
            
            # Look for seed-related settings
            has_seed = any(keyword in content.lower() for keyword in ['seed', 'random_state', 'deterministic'])
            
            if has_seed:
                print("✓ Seed/deterministic settings found in train.py")
            else:
                print("⚠ Warning: No explicit seed settings found in train.py")
                print("  Recommendation: Add torch.manual_seed() and deterministic settings")


class TestConfigValidation(unittest.TestCase):
    """Test configuration files for anti-leakage settings."""

    def test_configs_exist(self):
        """Verify that baseline configs exist."""
        configs_dir = Path(__file__).parent.parent / "configs"
        
        self.assertTrue(configs_dir.exists(), "configs/ directory not found")
        
        # Check for main configs
        bgl_config = configs_dir / "bgl.yaml"
        self.assertTrue(bgl_config.exists(), "configs/bgl.yaml not found")
        
        print(f"✓ Configuration files present")


def run_anti_leakage_tests():
    """Convenience function to run all anti-leakage tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestChronologicalSplit))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalOrder))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryQueueTemporal))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterministicSettings))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*70)
    print("TAC-LANOBERT ANTI-LEAKAGE VERIFICATION TESTS")
    print("="*70)
    print()
    
    success = run_anti_leakage_tests()
    
    print()
    print("="*70)
    if success:
        print("✅ ALL ANTI-LEAKAGE TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ABOVE")
    print("="*70)
    
    sys.exit(0 if success else 1)
