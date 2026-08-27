#!/bin/bash
# Prepare datasets for Kaggle Phase 4 upload

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "PREPARE KAGGLE DATASETS FOR PHASE 4"
echo "========================================================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
echo ""

if [ ! -d "outputs/BGL_lanobert/model/final" ]; then
    echo "❌ Phase 2 baseline not found: outputs/BGL_lanobert/"
    echo "   Please complete Phase 2 first."
    exit 1
fi

if [ ! -d "outputs/BGL_tac/model/final" ]; then
    echo "❌ Phase 3 TAC model not found: outputs/BGL_tac/"
    echo "   Please complete Phase 3 first."
    exit 1
fi

if [ ! -f "data/BGL/BGL_test_parsed.log" ]; then
    echo "⚠️  BGL preprocessed data not found: data/BGL/"
    echo "   You can still create BGL_lanobert.zip and BGL_tac.zip"
    echo "   But BGL_data.zip will not be created (slower on Kaggle)"
    echo ""
    BGL_DATA_READY=false
else
    BGL_DATA_READY=true
fi

echo "✅ Prerequisites check complete"
echo ""

# Create zips
echo "Creating dataset archives..."
echo ""

# 1. Phase 2 Baseline
if [ -f "BGL_lanobert.zip" ]; then
    echo "⚠️  BGL_lanobert.zip already exists. Overwrite? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Skipping BGL_lanobert.zip"
    else
        rm BGL_lanobert.zip
        echo "Creating BGL_lanobert.zip..."
        zip -r BGL_lanobert.zip outputs/BGL_lanobert/ -q
        echo "✅ BGL_lanobert.zip created ($(du -h BGL_lanobert.zip | cut -f1))"
    fi
else
    echo "Creating BGL_lanobert.zip..."
    zip -r BGL_lanobert.zip outputs/BGL_lanobert/ -q
    echo "✅ BGL_lanobert.zip created ($(du -h BGL_lanobert.zip | cut -f1))"
fi

echo ""

# 2. Phase 3 TAC
if [ -f "BGL_tac.zip" ]; then
    echo "⚠️  BGL_tac.zip already exists. Overwrite? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Skipping BGL_tac.zip"
    else
        rm BGL_tac.zip
        echo "Creating BGL_tac.zip..."
        zip -r BGL_tac.zip outputs/BGL_tac/ -q
        echo "✅ BGL_tac.zip created ($(du -h BGL_tac.zip | cut -f1))"
    fi
else
    echo "Creating BGL_tac.zip..."
    zip -r BGL_tac.zip outputs/BGL_tac/ -q
    echo "✅ BGL_tac.zip created ($(du -h BGL_tac.zip | cut -f1))"
fi

echo ""

# 3. BGL Preprocessed Data
if [ "$BGL_DATA_READY" = true ]; then
    if [ -f "BGL_data.zip" ]; then
        echo "⚠️  BGL_data.zip already exists. Overwrite? (y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            echo "Skipping BGL_data.zip"
        else
            rm BGL_data.zip
            echo "Creating BGL_data.zip..."
            zip -r BGL_data.zip data/BGL/ -q
            echo "✅ BGL_data.zip created ($(du -h BGL_data.zip | cut -f1))"
        fi
    else
        echo "Creating BGL_data.zip..."
        zip -r BGL_data.zip data/BGL/ -q
        echo "✅ BGL_data.zip created ($(du -h BGL_data.zip | cut -f1))"
    fi
else
    echo "⏭️  Skipping BGL_data.zip (preprocessed data not available)"
fi

echo ""
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
ls -lh *.zip 2>/dev/null | awk '{print "  " $9, "-", $5}'
echo ""
echo "Next steps:"
echo "  1. Go to https://www.kaggle.com/datasets"
echo "  2. Create new datasets and upload these zip files"
echo "  3. Attach datasets to your Phase 4 notebook"
echo ""
echo "See notebooks/PHASE4_README.md for detailed instructions."
echo "========================================================================"
