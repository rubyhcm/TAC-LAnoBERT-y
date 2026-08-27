#!/bin/bash
# Smoke Test for Phase 3: TAC-LAnoBERT Integration
# Tests that all TAC components work end-to-end

set -e  # Exit on error

echo "======================================================================"
echo "TAC-LAnoBERT Phase 3 Smoke Test"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        exit 1
    fi
}

# 1. Check TAC modules exist
echo "1. Checking TAC module files..."
for file in tac_lanobert/time2vec.py \
            tac_lanobert/time_delta.py \
            tac_lanobert/memory_queue.py \
            tac_lanobert/scoring.py \
            tac_lanobert/threshold.py \
            tac_lanobert/model.py \
            tac_lanobert/train_tac.py; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
        exit 1
    fi
done

# 2. Check config files exist
echo ""
echo "2. Checking config files..."
for cfg in configs/bgl_tac_full.yaml \
           configs/ablations/bgl_baseline.yaml \
           configs/ablations/bgl_time_only.yaml \
           configs/ablations/bgl_memory_only.yaml; do
    if [ -f "$cfg" ]; then
        echo "  ✓ $cfg exists"
    else
        echo "  ✗ $cfg missing"
        exit 1
    fi
done

# 3. Test Time2Vec module
echo ""
echo "3. Testing Time2Vec module..."
python -m tac_lanobert.time2vec > /tmp/time2vec_test.log 2>&1
print_status $? "Time2Vec unit test"

# 4. Test TimestampExtractor
echo ""
echo "4. Testing TimestampExtractor..."
python -m tac_lanobert.time_delta > /tmp/time_delta_test.log 2>&1
print_status $? "TimestampExtractor unit test"

# 5. Test Memory Queue
echo ""
echo "5. Testing Memory Queue..."
python -m tac_lanobert.memory_queue > /tmp/memory_test.log 2>&1
print_status $? "Memory Queue unit test"

# 6. Test TAC Model
echo ""
echo "6. Testing TAC Model (all modes)..."
python -m tac_lanobert.model > /tmp/model_test.log 2>&1
print_status $? "TAC Model unit test"

# 7. Test preprocess with timestamp extraction
echo ""
echo "7. Testing preprocess with timestamp extraction..."
if [ -f "data/BGL/BGL_train_normal.raw" ]; then
    # Create a small test file (first 100 lines)
    head -100 data/BGL/BGL_train_normal.raw > /tmp/test_small.raw
    
    python -m lanobert.preprocess \
        --config configs/bgl_tac_full.yaml \
        --split train \
        --in_path /tmp/test_small.raw \
        --out_path /tmp/test_small_parsed.txt \
        --extract_timestamps \
        > /tmp/preprocess_test.log 2>&1
    
    if [ -f "/tmp/test_small_parsed.timestamps" ]; then
        print_status 0 "Timestamp extraction in preprocess"
    else
        print_status 1 "Timestamp file not created"
    fi
else
    echo -e "${YELLOW}⚠${NC} Skipping (BGL data not found)"
fi

# 8. Test dataset loading with Time2Vec
echo ""
echo "8. Testing dataset with Time2Vec..."
python << EOF > /tmp/dataset_test.log 2>&1
import sys
sys.path.insert(0, '.')

from lanobert.tokenizer import load_tokenizer
from lanobert.dataset import LogLineDataset

# Create dummy data
with open('/tmp/test_dataset.txt', 'w') as f:
    for i in range(10):
        f.write(f"test log line {i}\n")

# Create dummy timestamps
with open('/tmp/test_dataset.timestamps', 'w') as f:
    for i in range(10):
        f.write(f"{1000000 + i * 60}\n")

# Test tokenizer (create minimal vocab)
import tempfile
vocab_dir = tempfile.mkdtemp()
from transformers import BertTokenizer
tok = BertTokenizer.from_pretrained('bert-base-uncased')
tok.save_pretrained(vocab_dir)

# Test dataset with Time2Vec
dataset = LogLineDataset(
    tokenizer=tok,
    file_path='/tmp/test_dataset.txt',
    max_len=128,
    use_time2vec=True,
    log_format='bgl'
)

# Check delta_t field
assert dataset.delta_t is not None, "delta_t not loaded"
assert len(dataset.delta_t) == 10, f"Expected 10 delta_t, got {len(dataset.delta_t)}"

# Check __getitem__
item = dataset[0]
assert 'delta_t' in item, "delta_t not in item dict"

print("✓ Dataset with Time2Vec works")
EOF

print_status $? "Dataset with Time2Vec loading"

# 9. Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}✓ Phase 3 Smoke Test PASSED${NC}"
echo "======================================================================"
echo ""
echo "Core components verified:"
echo "  ✓ Time2Vec layer"
echo "  ✓ Timestamp extraction"
echo "  ✓ Memory Queue"
echo "  ✓ TAC Model wrapper"
echo "  ✓ Preprocess integration"
echo "  ✓ Dataset integration"
echo ""
echo "Next steps:"
echo "  1. Run integration tests (tests/test_integration.py)"
echo "  2. Train 1 epoch on small subset"
echo "  3. Proceed to Phase 4 experiments"
echo ""
