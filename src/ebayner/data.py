
from __future__ import annotations
import os, re
import pandas as pd
from typing import List, Dict, Tuple
from .utils import read_tsv, whitespace_tokenize, bio_from_tag_sequence, allowed_labels_from_train

class DataModule:
    def __init__(self, listings_tsv: str, train_tagged_tsv: str):
        self.listings_tsv = listings_tsv
        self.train_tagged_tsv = train_tagged_tsv
        self._listings = None
        self._train = None

    @property
    def listings(self) -> pd.DataFrame:
        if self._listings is None:
            df = read_tsv(self.listings_tsv)
            df.columns = ["RecordNumber", "CategoryId", "Title"]
            df["RecordNumber"] = df["RecordNumber"].astype(int)
            df["CategoryId"] = df["CategoryId"].astype(int)
            self._listings = df
        return self._listings

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            df = read_tsv(self.train_tagged_tsv)
            df.columns = ["RecordNumber", "CategoryId", "Title", "Token", "Tag"]
            df["RecordNumber"] = df["RecordNumber"].astype(int)
            df["CategoryId"] = df["CategoryId"].astype(int)
            self._train = df
        return self._train

    def get_train_sequences(self) -> List[Dict]:
        rows = self.train
        seqs = []
        for rid, g in rows.groupby("RecordNumber", sort=True):
            toks = g["Token"].tolist()
            tags = g["Tag"].tolist()
            bio = bio_from_tag_sequence(tags)
            seqs.append({
                "record_id": rid,
                "category_id": int(g["CategoryId"].iloc[0]),
                "tokens": toks,
                "tags": tags,
                "bio_tags": bio,
                "title": g["Title"].iloc[0],
            })
        return seqs

    def get_quiz_superset(self) -> pd.DataFrame:
        qlo, qhi = 5001, 30000
        q = self.listings[(self.listings["RecordNumber"] >= qlo) & (self.listings["RecordNumber"] <= qhi)].copy()
        return q

    def get_listing_by_ids(self, record_ids: List[int]) -> pd.DataFrame:
        return self.listings[self.listings["RecordNumber"].isin(record_ids)].copy()
