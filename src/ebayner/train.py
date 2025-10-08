
from __future__ import annotations
import os, argparse, yaml, random
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from .data import DataModule
from .utils import ensure_dir, set_seed
from .gazetteer import build_gazetteers_from_train
from .model import train_token_classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(cfg["data"]["seed"])

    work = cfg["paths"]["work_dir"]
    ensure_dir(work)
    dm = DataModule(cfg["paths"]["listings_tsv"], cfg["paths"]["train_tagged_tsv"])

    seqs = dm.get_train_sequences()
    train_seqs, val_seqs = train_test_split(seqs, test_size=cfg["data"]["val_split"], random_state=cfg["data"]["seed"])

    # Build base gazetteers from train
    build_gazetteers_from_train(seqs, cfg["paths"]["gazetteers_dir"])

    # Train token classifier
    train_token_classifier(
        train_seqs, val_seqs,
        model_name=cfg["training"]["base_model_name"],
        max_length=cfg["data"]["max_length"],
        lr=cfg["training"]["lr"],
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        seed=cfg["data"]["seed"],
        work_dir=cfg["paths"]["model_dir"],
    )

if __name__ == "__main__":
    main()
