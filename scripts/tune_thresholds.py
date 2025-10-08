#!/usr/bin/env python3
from __future__ import annotations
import argparse, yaml, math
from copy import deepcopy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ebayner.data import DataModule
from ebayner.model import predict_tokens
from ebayner.utils import flatten_entities
from ebayner.evaluate import score_category
from ebayner.predict import decode_bio  # if you placed it here
from ebayner.postprocess import filter_and_gate

def eval_with(cfg, thresholds):
    dm = DataModule(cfg["paths"]["listings_tsv"], cfg["paths"]["train_tagged_tsv"])
    seqs = dm.get_train_sequences()
    _, val = train_test_split(seqs, test_size=cfg["data"]["val_split"], random_state=cfg["data"]["seed"])
    gold = {1:[],2:[]}; pred = {1:[],2:[]}
    for s in val:
        rid, cid = int(s["record_id"]), int(s["category_id"])
        tokens = s["tokens"]; bio = s["bio_tags"]
        gold[cid].extend((rid,n,v) for (n,v) in flatten_entities(tokens, bio))
        bio_pred, conf = predict_tokens(tokens, cfg["paths"]["model_dir"])
        spans = decode_bio(tokens, bio_pred, conf, None)
        aspects = filter_and_gate(cid, spans, thresholds)
        pred[cid].extend((rid,n,v) for (n,v) in aspects)
    beta = float(cfg.get("scoring",{}).get("beta",0.2))
    s1 = score_category(gold[1], pred[1], 1, beta); s2 = score_category(gold[2], pred[2], 2, beta)
    return (s1+s2)/2, s1, s2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # start from current thresholds or 0.80 default
    names = sorted({k for s in ["cat1","cat2"] for k in []})  # just placeholder
    # we’ll learn thresholds for any label we see during decode; coordinate ascent:
    cand = [0.60,0.70,0.75,0.80,0.85,0.90]
    thresholds = deepcopy(cfg.get("inference",{}).get("proba_thresholds",{}))
    best_score, *_ = eval_with(cfg, thresholds)

    improved = True
    while improved:
        improved = False
        # probe each label present by sampling from predictions of a small slice
        labels = set()
        dm = DataModule(cfg["paths"]["listings_tsv"], cfg["paths"]["train_tagged_tsv"])
        seqs = dm.get_train_sequences()[:400]  # small sample to discover labels quickly
        for s in seqs:
            bp, cf = predict_tokens(s["tokens"], cfg["paths"]["model_dir"])
            for lab in bp:
                if lab and lab != "O":
                    labels.add(lab[2:] if lab.startswith(("B-","I-")) else lab)
        for lab in labels:
            cur = thresholds.get(lab, 0.80)
            for t in cand:
                if abs(t-cur) < 1e-6: 
                    continue
                tmp = deepcopy(thresholds); tmp[lab] = t
                score, s1, s2 = eval_with(cfg, tmp)
                if score > best_score + 1e-5:
                    best_score = score; thresholds = tmp; improved = True
                    print(f"[improve] {lab}: {cur:.2f} -> {t:.2f}   score={score:.5f}")
                    break

    print("\nBest thresholds:", thresholds)
    score, s1, s2 = eval_with(cfg, thresholds)
    print(f"\nFinal Averaged Fβ=0.2: {score:.6f}  (cat1={s1:.6f}, cat2={s2:.6f})")
    # print a YAML block you can paste into configs
    print("\nYAML snippet:\n")
    print("inference:")
    print("  proba_thresholds:")
    for k,v in sorted(thresholds.items()):
        print(f"    {k}: {v:.2f}")

if __name__ == "__main__":
    main()
