from __future__ import annotations
import os, argparse, yaml, csv
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm

from .data import DataModule
from .utils import whitespace_tokenize, flatten_entities, save_submission, ensure_dir
from .model import predict_tokens
from .gazetteer import load_gazetteers, match_gazetteers
from .rules import rule_tag, validate

# -----------------------------
# Category-aware applicability
# -----------------------------
# Cat 1: Brake kits; Cat 2: Timing kits  (names taken from Annexure)
CAT_OK: Dict[int, set] = {
    1: {
        "Anzahl_Der_Einheiten","Besonderheiten","Bremsscheiben-Aussendurchmesser","Bremsscheibenart",
        "Einbauposition","Farbe","Größe","Hersteller","Herstellernummer","Herstellungsland_Und_-Region",
        "Im_Lieferumfang_Enthalten","Kompatible_Fahrzeug_Marke","Kompatibles_Fahrzeug_Jahr",
        "Kompatibles_Fahrzeug_Modell","Material","Maßeinheit","Modell","O","Oberflächenbeschaffenheit",
        "Oe/Oem_Referenznummer(N)","Produktart","Produktlinie","Stärke","Technologie"
    },
    2: {
        "Anwendung","Anzahl_Der_Einheiten","Besonderheiten","Einbauposition","Farbe","Größe","Hersteller",
        "Herstellernummer","Herstellungsland_Und_-Region","Im_Lieferumfang_Enthalten","Kompatible_Fahrzeug_Marke",
        "Kompatibles_Fahrzeug_Jahr","Kompatibles_Fahrzeug_Modell","Länge","Maßeinheit","Menge","Modell","O",
        "Oe/Oem_Referenznummer(N)","Produktart","Produktlinie","SAE_Viskosität","Technologie","Zähnezahl","Breite"
    },
}

def _allowed_for_category(cat_id: int, name: str) -> bool:
    ok = CAT_OK.get(int(cat_id))
    return (ok is None) or (name in ok)

# -----------------------------
# Precision-first merge
# -----------------------------
def merge_entities(
    tokens: List[str],
    model_bio: List[str],
    model_conf: List[float],
    cfg,
    category_id: int,
) -> List[Tuple[str, str]]:
    """
    Returns final (AspectName, AspectValue) pairs with:
      1) model spans filtered by per-aspect thresholds
      2) category gating (drop aspects not applicable to the item's category)
      3) rule-based + gazetteer additions (validated)
    """
    thresholds = cfg.get("inference", {}).get("proba_thresholds", {})
    default_thr = thresholds.get("default", 0.80)  # more conservative; β=0.2 favors precision

    # ---- 1) Model spans with mean confidence per span ----
    filtered: List[Tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tag = model_bio[i]
        if tag and tag.startswith("B-"):
            name = tag[2:]
            j = i + 1
            confs = [model_conf[i]]
            while j < len(tokens) and model_bio[j] == f"I-{name}":
                confs.append(model_conf[j])
                j += 1
            conf = sum(confs) / len(confs)
            value = " ".join(tokens[i:j]).strip()

            # category gate first
            if not _allowed_for_category(category_id, name):
                i = j
                continue

            thr = thresholds.get(name, default_thr)
            if conf >= thr and validate(name, value):
                filtered.append((name, value))
            i = j
        else:
            i += 1

    # ---- 2) Rules (high precision heuristics) ----
    ents_rules = rule_tag(tokens)
    # ---- 3) Gazetteers ----
    gaz = load_gazetteers(cfg["paths"]["gazetteers_dir"])
    ents_gaz = match_gazetteers(tokens, gaz)

    # Merge (union) with category-gating + validator
    all_ents = filtered[:]
    seen = set((n, v) for n, v in all_ents)

    for src in (ents_rules, ents_gaz):
        for n, v in src:
            if not _allowed_for_category(category_id, n):
                continue
            if validate(n, v) and (n, v) not in seen:
                all_ents.append((n, v))
                seen.add((n, v))

    return all_ents

# -----------------------------
# Submission block
# -----------------------------
def to_submission_block(df: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    # Expect columns from DataModule: RecordNumber, CategoryId, Title
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting", unit="title"):
        rid = int(row["RecordNumber"])
        cid = int(row["CategoryId"])
        title = row["Title"]
        tokens = title.split()  # literal whitespace tokenization
        bio, confs = predict_tokens(tokens, cfg["paths"]["model_dir"])
        ents = merge_entities(tokens, bio, confs, cfg, category_id=cid)
        for name, value in ents:
            rows.append({
                "RecordNumber": rid,
                "CategoryId": cid,
                "AspectName": name,
                "AspectValue": value
            })
    return pd.DataFrame(rows, columns=["RecordNumber","CategoryId","AspectName","AspectValue"])

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--quiz", action="store_true", help="Predict for quiz superset")
    parser.add_argument("--to-submission", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    dm = DataModule(cfg["paths"]["listings_tsv"], cfg["paths"]["train_tagged_tsv"])
    if args.quiz:
        df = dm.get_quiz_superset()    # your existing helper
    else:
        # small sample by default when not --quiz (so you can preview quickly)
        df = dm.listings.head(100).copy()

    sub = to_submission_block(df, cfg)
    ensure_dir(cfg["paths"]["work_dir"])
    if args.to_submission:
        save_submission(
            sub[["RecordNumber","CategoryId","AspectName","AspectValue"]],
            cfg["paths"]["submission_path"]
        )
        print(f"Wrote submission to {cfg['paths']['submission_path']}")
    else:
        print(sub.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
