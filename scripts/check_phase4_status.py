#!/usr/bin/env python3
"""
Phase 4 Status Summary

Kiểm tra trạng thái hoàn thành của Phase 4 experiments.
"""

import json
from pathlib import Path


def check_file_exists(path):
    """Check if file exists and return status emoji."""
    return "✅" if Path(path).exists() else "❌"


def load_json(path):
    """Load JSON file if exists."""
    if Path(path).exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def main():
    print("=" * 80)
    print("Phase 4: Main Experiments - Status Summary")
    print("=" * 80)
    
    # Check E1
    print("\n📋 E1: Baseline Verification")
    print("-" * 80)
    e1_report = "outputs/BGL_lanobert/results/e1_baseline_verification_report.json"
    e1_status = check_file_exists(e1_report)
    print(f"{e1_status} E1 Report: {e1_report}")
    
    e1_data = load_json(e1_report)
    if e1_data:
        comp = e1_data.get('comparison', {})
        print(f"   Status: {comp.get('status', 'UNKNOWN')}")
        print(f"   F1 diff: {comp.get('f1_diff_pct', 0):.4f}%")
        print(f"   AUROC diff: {comp.get('auroc_diff_pct', 0):.4f}%")
    
    # Check E2
    print("\n📋 E2: Main Comparison")
    print("-" * 80)
    
    # Check if training complete
    model_path = "outputs/BGL_tac/model/final"
    model_status = check_file_exists(model_path)
    print(f"{model_status} TAC Model Trained: {model_path}")
    
    # Check scores
    scores = [
        "outputs/BGL_tac/results/scores_tac_mlm_error.npy",
        "outputs/BGL_tac/results/scores_tac_mahalanobis.npy",
        "outputs/BGL_tac/results/scores_tac_hybrid.npy"
    ]
    for score_file in scores:
        status = check_file_exists(score_file)
        print(f"{status} {Path(score_file).name}")
    
    # Check E2 report
    e2_report = "outputs/BGL_tac/results/e2_main_comparison_report.json"
    e2_status = check_file_exists(e2_report)
    print(f"{e2_status} E2 Report: {e2_report}")
    
    e2_data = load_json(e2_report)
    if e2_data:
        improvements = e2_data.get('improvements', {})
        print(f"\n   Improvements:")
        print(f"   - F1 Δ:    {improvements.get('f1_delta_pct', 0):+.2f}%")
        print(f"   - AUROC Δ: {improvements.get('auroc_delta_pct', 0):+.2f}%")
        print(f"   - FPR Δ:   {improvements.get('fpr_delta_pct', 0):+.2f}%")
        
        fpr_delta = improvements.get('fpr_delta_pct', 0)
        if fpr_delta <= -15:
            print(f"   ✅ H1 MET: FPR reduced ≥15%")
        else:
            print(f"   ⚠️  H1 NOT MET: FPR reduced only {fpr_delta:.2f}%")
    
    # Check E3
    print("\n📋 E3: Early Detection Test")
    print("-" * 80)
    
    e3_reports = [
        "outputs/BGL_tac/results/e3_early_detection_mlm_report.json",
        "outputs/BGL_tac/results/e3_early_detection_mahalanobis_report.json",
        "outputs/BGL_tac/results/e3_early_detection_hybrid_report.json"
    ]
    
    e3_complete = True
    for report in e3_reports:
        status = check_file_exists(report)
        score_type = Path(report).stem.split('_')[-2]
        print(f"{status} E3 Report ({score_type}): {report}")
        
        if status == "❌":
            e3_complete = False
        
        data = load_json(report)
        if data:
            dlt = data.get('dlt_results', {})
            print(f"   - Mean DLT: {dlt.get('mean_dlt', 0)/60:.2f} min")
            print(f"   - EWR (≥5min): {dlt.get('early_warning_rate', 0):.2f}%")
    
    # Overall Status
    print("\n" + "=" * 80)
    print("📊 Phase 4 Overall Status")
    print("=" * 80)
    
    e1_done = e1_data is not None
    e2_done = e2_data is not None
    e3_done = e3_complete
    
    print(f"E1 (Baseline Verification):  {'✅ COMPLETE' if e1_done else '❌ INCOMPLETE'}")
    print(f"E2 (Main Comparison):         {'✅ COMPLETE' if e2_done else '❌ INCOMPLETE'}")
    print(f"E3 (Early Detection):         {'✅ COMPLETE' if e3_done else '❌ INCOMPLETE'}")
    
    if e1_done and e2_done and e3_done:
        print("\n🎉 Phase 4: COMPLETE!")
        print("\nNext Steps:")
        print("  1. Review E2 comparison results")
        print("  2. Review E3 DLT distribution")
        print("  3. Proceed to Phase 5: bash scripts/run_phase5.sh")
    else:
        print("\n⚠️  Phase 4: INCOMPLETE")
        print("\nTo complete:")
        if not e1_done:
            print("  - Run E1: python -m experiments.run_baseline")
        if not e2_done:
            print("  - Run E2: python -m experiments.run_main")
        if not e3_done:
            print("  - Run E3: python -m experiments.run_early_detection --score-type [mlm|mahalanobis|hybrid]")
        print("\nOr run all: bash scripts/run_phase4.sh")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
