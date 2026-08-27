"""
Time Delta Extraction: Extract timestamps and compute Δt from log lines.

Supports multiple timestamp formats for BGL, Thunderbird, HDFS.
"""

import re
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np


class TimestampExtractor:
    """
    Extract timestamps from log lines and compute time deltas.
    
    Supported formats:
    - BGL: "1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ..."
           Unix timestamp at start
    - Thunderbird: "- 1129903038 0 master.HPC ..."
           Unix timestamp at position 1
    - HDFS: "081109 203518 148 INFO ..."
           YYMMDD HHMMSS format
    """
    
    # Regex patterns for different log formats
    PATTERNS = {
        'bgl': r'^-?\s*(\d{10})\s',                # Unix timestamp at start (optional -)
        'thunderbird': r'^-\s+(\d{10})\s',         # Unix timestamp after "-"
        'hdfs': r'^(\d{6})\s+(\d{6})',             # YYMMDD HHMMSS
    }
    
    def __init__(self, log_format: str = 'bgl'):
        """
        Args:
            log_format: One of ['bgl', 'thunderbird', 'hdfs']
        """
        self.log_format = log_format.lower()
        if self.log_format not in self.PATTERNS:
            raise ValueError(f"Unsupported log format: {log_format}. "
                           f"Choose from {list(self.PATTERNS.keys())}")
        
        self.pattern = re.compile(self.PATTERNS[self.log_format])
        self.last_timestamp = None  # For computing Δt
    
    def extract_timestamp(self, log_line: str) -> Optional[float]:
        """
        Extract timestamp from a log line.
        
        Args:
            log_line: Raw log line string
        
        Returns:
            Unix timestamp (seconds since epoch) or None if not found
        """
        match = self.pattern.search(log_line)
        if not match:
            return None
        
        if self.log_format in ['bgl', 'thunderbird']:
            # Unix timestamp (already in seconds)
            return float(match.group(1))
        
        elif self.log_format == 'hdfs':
            # YYMMDD HHMMSS format
            date_str = match.group(1)  # YYMMDD
            time_str = match.group(2)  # HHMMSS
            
            # Parse: assume 20XX century for YY
            try:
                year = int('20' + date_str[0:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                hour = int(time_str[0:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                
                dt = datetime(year, month, day, hour, minute, second)
                return dt.timestamp()
            except ValueError:
                return None
        
        return None
    
    def compute_delta_t(self, current_timestamp: Optional[float]) -> float:
        """
        Compute time delta from last timestamp.
        
        Args:
            current_timestamp: Current Unix timestamp (seconds)
        
        Returns:
            Δt in milliseconds. Returns 0 for first event or if timestamp is None.
        """
        if current_timestamp is None:
            return 0.0
        
        if self.last_timestamp is None:
            # First event
            self.last_timestamp = current_timestamp
            return 0.0
        
        # Compute delta in milliseconds
        delta_t_ms = (current_timestamp - self.last_timestamp) * 1000.0
        
        # Update last timestamp
        self.last_timestamp = current_timestamp
        
        # Clamp negative deltas to 0 (for out-of-order logs)
        return max(0.0, delta_t_ms)
    
    def reset(self):
        """Reset state for new session/file."""
        self.last_timestamp = None
    
    @staticmethod
    def normalize_delta_t(delta_t_ms: float) -> float:
        """
        Normalize time delta using log transformation.
        
        Δt_norm = log(1 + Δt_ms)
        
        This stabilizes numerical range:
        - 0ms → 0
        - 1ms → 0.69
        - 10ms → 2.40
        - 100ms → 4.62
        - 1s (1000ms) → 6.91
        - 1min (60000ms) → 11.00
        - 1h (3600000ms) → 15.09
        
        Args:
            delta_t_ms: Raw time delta in milliseconds
        
        Returns:
            Normalized delta_t for Time2Vec input
        """
        return np.log1p(max(0.0, delta_t_ms))  # clip to 0 to avoid NaN on negative delta


def extract_timestamps_from_file(
    file_path: str,
    log_format: str = 'bgl',
    max_lines: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Extract timestamps and compute deltas from a log file.
    
    Args:
        file_path: Path to log file
        log_format: Log format ('bgl', 'thunderbird', 'hdfs')
        max_lines: Maximum lines to process (None = all)
    
    Returns:
        (timestamps, delta_t_normalized): Both as lists of floats
    """
    extractor = TimestampExtractor(log_format=log_format)
    
    timestamps = []
    delta_t_list = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Extract timestamp
            ts = extractor.extract_timestamp(line)
            
            # Compute delta_t
            delta_t_ms = extractor.compute_delta_t(ts)
            delta_t_norm = TimestampExtractor.normalize_delta_t(delta_t_ms)
            
            timestamps.append(ts if ts is not None else 0.0)
            delta_t_list.append(delta_t_norm)
    
    return timestamps, delta_t_list


# Unit test
def _test_timestamp_extraction():
    """Test timestamp extraction on sample log lines."""
    
    # BGL sample
    bgl_line = "1117838570 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
    extractor_bgl = TimestampExtractor('bgl')
    ts_bgl = extractor_bgl.extract_timestamp(bgl_line)
    assert ts_bgl == 1117838570, f"BGL timestamp mismatch: {ts_bgl}"
    
    # Thunderbird sample
    tbird_line = "- 1129903038 0 master.HPC InstallMode set"
    extractor_tbird = TimestampExtractor('thunderbird')
    ts_tbird = extractor_tbird.extract_timestamp(tbird_line)
    assert ts_tbird == 1129903038, f"Thunderbird timestamp mismatch: {ts_tbird}"
    
    # Test delta_t computation
    extractor = TimestampExtractor('bgl')
    extractor.reset()
    
    dt1 = extractor.compute_delta_t(1000.0)  # First event
    assert dt1 == 0.0, "First delta_t should be 0"
    
    dt2 = extractor.compute_delta_t(1001.5)  # 1.5 seconds later = 1500ms
    assert dt2 == 1500.0, f"Expected 1500ms, got {dt2}"
    
    # Test normalization
    norm_0 = TimestampExtractor.normalize_delta_t(0.0)
    assert abs(norm_0 - 0.0) < 1e-6, "log(1+0) should be 0"
    
    norm_1000 = TimestampExtractor.normalize_delta_t(1000.0)  # 1 second
    expected = np.log1p(1000.0)  # ~6.91
    assert abs(norm_1000 - expected) < 1e-6, f"Normalization mismatch: {norm_1000} vs {expected}"
    
    print("✅ Timestamp extraction tests passed!")
    print(f"   BGL timestamp: {ts_bgl}")
    print(f"   Thunderbird timestamp: {ts_tbird}")
    print(f"   Delta_t test: {dt2}ms")
    print(f"   Normalized 1s: {norm_1000:.4f}")


if __name__ == "__main__":
    _test_timestamp_extraction()
