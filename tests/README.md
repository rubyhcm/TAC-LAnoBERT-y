# Tests Directory

Unit tests và verification tools cho TAC-LAnoBERT project.

## Files

### `test_data_leakage.py`
Anti-leakage verification tests - đảm bảo không có data leakage trong quá trình training/testing.

**Tests included:**
1. `TestChronologicalSplit` - Verify chronological split integrity
2. `TestTemporalOrder` - Verify timestamps are monotonic  
3. `TestMemoryQueueTemporal` - Verify FIFO queue design (Phase 3 prep)
4. `TestDeterministicSettings` - Verify reproducibility settings
5. `TestConfigValidation` - Verify configuration files

**Run:**
```bash
python3 tests/test_data_leakage.py
```

**Expected output:**
```
======================================================================
TAC-LANOBERT ANTI-LEAKAGE VERIFICATION TESTS
======================================================================

✓ BGL split verified: train_normal=3,496,193, test_total=1,251,770
✓ No shuffle=True found in inference DataLoader
✓ Thunderbird log accessible
✓ Memory Queue structure verified (FIFO design)
✓ Seed/deterministic settings found in train.py
✓ Configuration files present

======================================================================
✅ ALL ANTI-LEAKAGE TESTS PASSED
======================================================================
```

### `verify_timestamps.py`
BGL timestamp extraction và validation tool.

**Features:**
- Extract Unix timestamps from raw logs
- Verify monotonic ordering
- Compute time delta (Δt) statistics
- Detect duplicate timestamps
- Time range analysis

**Run:**
```bash
python3 tests/verify_timestamps.py
```

**Output example:**
```
BGL TEST SET TIMESTAMP VERIFICATION
======================================================================
✓ Extracted 100,000 timestamps
✓ Timestamps are monotonically non-decreasing

Time Range:
  Start: 2005-11-04 06:23:59
  End:   2005-11-04 23:59:43
  Duration: 17.60 hours

Time Deltas (Δt) Statistics:
  Mean: 12.089 seconds
  Median: 1.000 seconds
  Min: 1.000000 seconds
  Max: 6315.000 seconds
  P95: 1.000s, P99: 1.000s

⚠ Found 94,759 events with identical timestamp
  (Normal for high-frequency logging systems)
```

## Usage

### Run all tests
```bash
# Comprehensive verification (recommended)
bash scripts/verify_phase1.sh

# Individual test suites
python3 tests/test_data_leakage.py
python3 tests/verify_timestamps.py
```

### Run specific test
```bash
# Single test class
python3 -m unittest tests.test_data_leakage.TestChronologicalSplit

# Single test method
python3 -m unittest tests.test_data_leakage.TestChronologicalSplit.test_bgl_split_chronological
```

## Adding New Tests

When adding tests for Phase 3+ modules:

1. **Create test file**: `test_<module_name>.py`
2. **Import module**: Add parent directory to path
3. **Follow naming**: `TestXxx` classes, `test_xxx` methods
4. **Add to suite**: Update `verify_phase1.sh` or create phase-specific script

Example structure:
```python
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tac_lanobert.time2vec import Time2VecLayer

class TestTime2Vec(unittest.TestCase):
    def test_output_shape(self):
        # Test implementation
        pass

if __name__ == '__main__':
    unittest.main()
```

## Phase-Specific Tests

### Phase 1 (Complete ✅)
- [x] `test_data_leakage.py` - 7/7 passing
- [x] `verify_timestamps.py` - Passing

### Phase 3 (Planned)
- [ ] `test_time2vec.py` - Time2Vec layer
- [ ] `test_memory_queue.py` - FIFO queue operations
- [ ] `test_welford.py` - Online statistics
- [ ] `test_mahalanobis.py` - Distance computation
- [ ] `test_scoring.py` - Hybrid scoring

### Phase 4+ (Planned)
- [ ] `test_integration.py` - End-to-end pipeline
- [ ] `test_performance.py` - Latency benchmarks

## CI/CD Integration

For future CI/CD setup:
```bash
# In .github/workflows/tests.yml or similar
- name: Run Phase 1 Tests
  run: |
    python3 tests/test_data_leakage.py
    python3 tests/verify_timestamps.py
    bash scripts/verify_phase1.sh
```

## Test Data

Tests use existing datasets in `data/`:
- `data/BGL/BGL_test.raw` - For timestamp verification
- `data/BGL/split_stats.json` - For split verification
- `data/Thunderbird/Thunderbird.log` - For accessibility check

No additional test fixtures required for Phase 1.
