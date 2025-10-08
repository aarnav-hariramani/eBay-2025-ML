
from __future__ import annotations
import os, re, json, random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Iterable
from dataclasses import dataclass

RNG = np.random.default_rng(42)

def set_seed(seed: int = 42):
    import torch, random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_tsv(path: str) -> pd.DataFrame:
    # Annexure: TSV, CSV-encoded, UTF-8, keep blanks
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=None, dtype=str)

def save_submission(df: pd.DataFrame, path: str):
    import csv
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)

def whitespace_tokenize(title: str) -> List[str]:
    toks = [t for t in title.split() if t != ""]
    return toks

def bio_from_tag_sequence(tags: List[str]) -> List[str]:
    """
    Annexure rule:
    - empty tag => continuation of previous entity (I-<prev>)
    - same non-empty tag repeated in consecutive rows => different entities => B- for each
    """
    bio = []
    last_non_empty = None
    for t in tags:
        if t == "":
            if last_non_empty is None:
                bio.append("O")
            else:
                bio.append(f"I-{last_non_empty}")
        else:
            bio.append(f"B-{t}")
            last_non_empty = t
    return bio

def flatten_entities(tokens: List[str], bio_tags: List[str]) -> List[Tuple[str, str]]:
    """
    Convert tokens + BIO tags into (aspect_name, aspect_value) pairs.
    We treat each 'B-x' as a new entity and consume following 'I-x' tokens.
    """
    out = []
    i = 0
    while i < len(tokens):
        tag = bio_tags[i]
        if tag.startswith("B-"):
            name = tag[2:]
            val = [tokens[i]]
            j = i + 1
            while j < len(tokens) and bio_tags[j] == f"I-{name}":
                val.append(tokens[j])
                j += 1
            out.append((name, " ".join(val)))
            i = j
        else:
            i += 1
    return out

def allowed_labels_from_train(train_df: pd.DataFrame) -> List[str]:
    labels = sorted([x for x in train_df["Tag"].unique() if x not in ("", "O")])
    return labels

def quiz_id_range() -> Tuple[int, int]:
    # Annexure: quiz is 5,001..30,000 of Listing_Titles.tsv
    return (5001, 30000)
