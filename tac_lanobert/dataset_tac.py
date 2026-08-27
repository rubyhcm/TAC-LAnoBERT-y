"""TAC-LAnoBERT Extension for Dataset.

Independent dataset class that handles text tokenization and timestamps 
for Time2Vec without relying on the baseline lanobert package.
"""
from __future__ import annotations

import os
from typing import List, Optional

from torch.utils.data import Dataset
from tac_lanobert.time_delta import TimestampExtractor


class TACLogLineDataset(Dataset):
    """One normalized log line -> one tokenized example (no NSP) + delta_t.

    Args:
        tokenizer: a fast BERT tokenizer.
        file_path: path to a newline-delimited normalized corpus.
        max_len: max sequence length (longer lines are truncated).
        skip_empty: drop blank lines.
        log_format: log format for timestamp extraction ('bgl', 'thunderbird', 'hdfs')
    """

    def __init__(
        self, 
        tokenizer, 
        file_path: str, 
        max_len: int = 512, 
        skip_empty: bool = True,
        log_format: str = "bgl",
        load_timestamps: bool = True,
    ):
        assert os.path.isfile(file_path), f"Input file not found: {file_path}"
        self.max_len = max_len
        self.log_format = log_format
        self.delta_t: Optional[List[float]] = None

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        if skip_empty:
            lines = [ln for ln in lines if ln]

        print(f"[tac_dataset] pre-tokenizing {len(lines):,} lines...")
        batch_enc = tokenizer(
            lines,
            truncation=True,
            max_length=max_len,
            return_special_tokens_mask=True,
        )
        self.input_ids: List[List[int]] = batch_enc["input_ids"]
        self.attention_mask: List[List[int]] = batch_enc["attention_mask"]
        self.special_tokens_mask: List[List[int]] = batch_enc["special_tokens_mask"]
        print(f"[tac_dataset] pre-tokenization done.")
        
        # Load timestamps if we have data and it is requested
        if load_timestamps and len(self.input_ids) > 0:
            self._load_timestamps(file_path, log_format)

    def _load_timestamps(self, file_path: str, log_format: str):
        """Load timestamps and compute delta_t."""
        base_path = os.path.splitext(file_path)[0]
        timestamp_path = base_path + ".timestamps"
        
        if os.path.isfile(timestamp_path):
            print(f"[tac_dataset] loading timestamps from {timestamp_path}")
            with open(timestamp_path, "r", encoding="utf-8") as f:
                timestamps = [float(line.strip()) for line in f if line.strip()]
            
            if len(timestamps) != len(self.input_ids):
                print(f"[tac_dataset] WARNING: timestamp count mismatch: "
                      f"{len(timestamps)} vs {len(self.input_ids)} lines")
                if len(timestamps) < len(self.input_ids):
                    timestamps.extend([0.0] * (len(self.input_ids) - len(timestamps)))
                else:
                    timestamps = timestamps[:len(self.input_ids)]
        else:
            print(f"[tac_dataset] .timestamps file not found, extracting on-the-fly...")
            raw_path = file_path.replace("_normal.txt", "_raw.txt").replace("_log.txt", "_raw.txt").replace("_normal_parsed.log", "_normal.raw").replace("_parsed.log", ".raw")
            
            if not os.path.isfile(raw_path):
                print(f"[tac_dataset] WARNING: raw file not found at {raw_path}, using zero delta_t")
                timestamps = [0.0] * len(self.input_ids)
            else:
                from tac_lanobert.time_delta import extract_timestamps_from_file
                timestamps, _ = extract_timestamps_from_file(raw_path, log_format=log_format)
                
                if len(timestamps) != len(self.input_ids):
                    if len(timestamps) < len(self.input_ids):
                        timestamps.extend([0.0] * (len(self.input_ids) - len(timestamps)))
                    else:
                        timestamps = timestamps[:len(self.input_ids)]
        
        extractor = TimestampExtractor(log_format=log_format)
        
        delta_t_list = []
        for ts in timestamps:
            delta_ms = extractor.compute_delta_t(ts if ts > 0 else None)
            delta_norm = TimestampExtractor.normalize_delta_t(delta_ms)
            delta_t_list.append(delta_norm)
        
        self.delta_t = delta_t_list
        print(f"[tac_dataset] computed {len(self.delta_t)} delta_t values (range: "
              f"{min(self.delta_t):.4f} to {max(self.delta_t):.4f})")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "special_tokens_mask": self.special_tokens_mask[idx],
        }
        
        if self.delta_t is not None:
            seq_len = len(item["input_ids"])
            item["delta_t"] = [self.delta_t[idx]] * seq_len
            
        return item


def read_lines(file_path: str, limit: int | None = None) -> List[str]:
    """Utility: read normalized lines from a corpus, optionally capped at `limit`."""
    out: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(ln)
            if limit is not None and len(out) >= limit:
                break
    return out
