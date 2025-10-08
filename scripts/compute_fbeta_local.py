#!/usr/bin/env python3
"""
Compute local weighted F_{beta=0.2} on the same validation split that train.py uses.
- Uses the trained model from cfg['paths']['model_dir']
- Uses the same merge (model + rules + gazetteers) as predict.py
- Reproduces the competition metric (per-category weighted F_beta then averaged)

Run:
  python scripts/compute_fbeta_local.py --config configs/default.yaml
"""

from __future__ import annotations
import argparse
import yaml
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

from ebayner.data import DataModule
from ebayner.utils import flatten_entities
from ebayner.model import predict_tokens
from ebayner.evaluate import score_category
from ebayner.predict import merge_entities  # reuse your hybrid merger


def seq_entities_gold(seqs):
    """Extract gold (name, value) entity pairs from the BIO tags in seqs."""
    gold_by_cat = {1: [], 2: []}
    for s in seqs:
        rid = int(s["record_id"])
        cid = int(s["category_id"])
        ents = flatten_entities(s["tokens"], s["bio_tags"])  # -> [(name, value)]
        gold_by_cat[cid].extend((rid, n, v) for (n, v) in ents)
    return gold_by_cat


def seq_entities_pred(seqs, cfg, max_samples=None):
    """Predict entities with the trained model and your hybrid merge."""
    pred_by_cat = {1: [], 2: []}

    # Optionally test on subset to avoid long runs
    if max_samples:
        seqs = seqs[:max_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[INFO] Predicting {len(seqs)} samples on device={device}...\n")

    for s in tqdm(seqs, desc="Predicting", unit="title"):
        rid = int(s["record_id"])
        cid = int(s["category_id"])
        tokens = s["tokens"]

        # 1) model token predictions
        bio_pred, confs = predict_tokens(tokens, cfg["paths"]["model_dir"])

        # 2) hybrid merge (model -> thresholds -> rules -> gazetteers)
        ents = merge_entities(tokens, bio_pred, confs, cfg)

        # 3) accumulate per category
        pred_by_cat[cid].extend((rid, n, v) for (n, v) in ents)

    return pred_by_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", type=int, default=None,
                    help="Optionally limit to N samples for quick validation (debug mode).")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # Load data and produce the SAME split train.py used
    dm = DataModule(cfg["paths"]["listings_tsv"], cfg["paths"]["train_tagged_tsv"])
    seqs = dm.get_train_sequences()
    train_seqs, val_seqs = train_test_split(
        seqs,
        test_size=cfg["data"]["val_split"],
        random_state=cfg["data"]["seed"],
    )

    # Build gold and predictions on the validation split only
    gold = seq_entities_gold(val_seqs)
    pred = seq_entities_pred(val_seqs, cfg, max_samples=args.subset)

    beta = float(cfg.get("scoring", {}).get("beta", 0.2))

    # Compute per-category scores (competition uses average of the two)
    s1 = score_category(gold[1], pred[1], category_id=1, beta=beta)
    s2 = score_category(gold[2], pred[2], category_id=2, beta=beta)
    final = (s1 + s2) / 2.0

    print("\n================ Local Validation (competition metric) ================\n")
    print(f"  Category 1 F_beta={beta}: {s1:.6f}")
    print(f"  Category 2 F_beta={beta}: {s2:.6f}")
    print("  -----------------------------------------------------------------")
    print(f"  Averaged F_beta={beta}:   {final:.6f}")
    print("\n(These numbers approximate the leaderboard behavior, but are computed on your\n"
          "validation split, not on EvalAI's hidden quiz/test sets.)\n")


if __name__ == "__main__":
    main()
