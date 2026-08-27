#!/usr/bin/env python3
"""
Verify BGL timestamp extraction and chronological order.

This script validates that:
1. Timestamps can be extracted from raw BGL logs
2. Timestamps are monotonically non-decreasing
3. Time deltas (Δt) can be computed
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

def analyze_bgl_timestamps(file_path, max_lines=10000):
    """Analyze BGL raw log timestamps."""
    print(f"Analyzing: {file_path}")
    print("="*70)
    
    timestamps = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # BGL format: label timestamp date ...
                    ts = int(parts[1])
                    timestamps.append(ts)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse line {i+1}")
                    continue
    
    if not timestamps:
        print("❌ No timestamps extracted!")
        return False
    
    print(f"✓ Extracted {len(timestamps):,} timestamps")
    
    # Convert to numpy for analysis
    ts_array = np.array(timestamps)
    
    # Check monotonicity
    diffs = np.diff(ts_array)
    negative_diffs = diffs < 0
    
    if negative_diffs.any():
        print(f"❌ Found {negative_diffs.sum()} negative time jumps!")
        # Show first few violations
        violations = np.where(negative_diffs)[0][:5]
        for idx in violations:
            print(f"   Position {idx}: {timestamps[idx]} → {timestamps[idx+1]} "
                  f"(Δt = {timestamps[idx+1] - timestamps[idx]})")
        return False
    
    print(f"✓ Timestamps are monotonically non-decreasing")
    
    # Time range analysis
    t_start = timestamps[0]
    t_end = timestamps[-1]
    dt_start = datetime.fromtimestamp(t_start)
    dt_end = datetime.fromtimestamp(t_end)
    duration_hours = (t_end - t_start) / 3600
    
    print(f"\nTime Range:")
    print(f"  Start: {dt_start} (epoch: {t_start})")
    print(f"  End:   {dt_end} (epoch: {t_end})")
    print(f"  Duration: {duration_hours:.2f} hours ({duration_hours/24:.2f} days)")
    
    # Delta analysis
    deltas = diffs[diffs > 0]  # Only non-zero deltas
    if len(deltas) > 0:
        print(f"\nTime Deltas (Δt) Statistics:")
        print(f"  Mean: {deltas.mean():.3f} seconds")
        print(f"  Median: {np.median(deltas):.3f} seconds")
        print(f"  Min: {deltas.min():.6f} seconds")
        print(f"  Max: {deltas.max():.3f} seconds")
        print(f"  Std: {deltas.std():.3f} seconds")
        
        # Percentiles
        p25, p75, p95, p99 = np.percentile(deltas, [25, 75, 95, 99])
        print(f"  P25: {p25:.3f}s, P75: {p75:.3f}s, P95: {p95:.3f}s, P99: {p99:.3f}s")
    
    # Check for duplicate timestamps (events at same time)
    zero_deltas = (diffs == 0).sum()
    if zero_deltas > 0:
        print(f"\n⚠ Found {zero_deltas:,} events with identical timestamp to previous event")
        print(f"  This is normal for high-frequency logging systems")
    
    return True


def main():
    # Check BGL test set
    bgl_test = Path(__file__).parent.parent / "data" / "BGL" / "BGL_test.raw"
    
    if not bgl_test.exists():
        print(f"❌ File not found: {bgl_test}")
        return 1
    
    print("BGL TEST SET TIMESTAMP VERIFICATION")
    print("="*70)
    success = analyze_bgl_timestamps(bgl_test, max_lines=100000)
    
    if success:
        print("\n" + "="*70)
        print("✅ BGL TIMESTAMP VERIFICATION PASSED")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("❌ BGL TIMESTAMP VERIFICATION FAILED")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
