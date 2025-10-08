
from __future__ import annotations
import os, argparse, yaml, csv
from typing import List, Tuple, Dict
import pandas as pd
from .data import DataModule
from .utils import whitespace_tokenize, flatten_entities, save_submission, ensure_dir
from .model import predict_tokens
from .gazetteer import load_gazetteers, match_gazetteers
from .rules import rule_tag, validate

def merge_entities(tokens: List[str], model_bio: List[str], model_conf: List[float], cfg) -> List[Tuple[str,str]]:
    # 1) model entities with thresholds
    ents_model = flatten_entities(tokens, model_bio)
    thresholds = cfg["inference"]["proba_thresholds"]
    filtered = []
    i = 0
    while i < len(tokens):
        tag = model_bio[i]
        if tag.startswith("B-"):
            name = tag[2:]
            thr = thresholds.get(name, thresholds.get("default", 0.5))
            confs = [model_conf[i]]
            j = i+1
            while j < len(tokens) and model_bio[j] == f"I-{name}":
                confs.append(model_conf[j])
                j += 1
            conf = sum(confs)/len(confs)
            value = " ".join(tokens[i:j])
            if conf >= thr and validate(name, value):
                filtered.append((name, value))
            i = j
        else:
            i += 1

    # 2) rules
    ents_rules = rule_tag(tokens)

    # 3) gazetteers
    gaz = load_gazetteers(cfg["paths"]["gazetteers_dir"])
    ents_gaz = match_gazetteers(tokens, gaz)

    # Merge (union + validator)
    all_ents = filtered[:]
    seen = set((n,v) for n,v in all_ents)

    for src in (ents_rules, ents_gaz):
        for n,v in src:
            if validate(n,v):
                if (n,v) not in seen:
                    all_ents.append((n,v))
                    seen.add((n,v))
    return all_ents

def to_submission_block(df: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rid = int(row["RecordNumber"])
        cid = int(row["CategoryId"])
        title = row["Title"]
        tokens = title.split()
        bio, confs = predict_tokens(tokens, cfg["paths"]["model_dir"])
        ents = merge_entities(tokens, bio, confs, cfg)
        for name, value in ents:
            rows.append({"RecordNumber": rid, "CategoryId": cid, "AspectName": name, "AspectValue": value})
    return pd.DataFrame(rows, columns=["RecordNumber","CategoryId","AspectName","AspectValue"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--quiz", action="store_true", help="Predict for quiz superset (5,001..30,000)")
    parser.add_argument("--to-submission", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    dm = DataModule(cfg["paths"]["listings_tsv"], cfg["paths"]["train_tagged_tsv"])
    if args.quiz:
        df = dm.get_quiz_superset()
    else:
        df = dm.listings.head(100).copy()

    sub = to_submission_block(df, cfg)
    ensure_dir(cfg["paths"]["work_dir"])
    if args.to_submission:
        save_submission(sub[["RecordNumber","CategoryId","AspectName","AspectValue"]], cfg["paths"]["submission_path"])
        print(f"Wrote submission to {cfg['paths']['submission_path']}")
    else:
        print(sub.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
