
# eBay ML 2025 NER — Hybrid Precision-First System (German Motors)

This repo is a production-ready, precision-first pipeline for the eBay University ML 2025 challenge.
It follows the Annexure exactly (whitespace tokenization, TSV/CSV-encoded inputs, submission format)
and implements a hybrid model:

1) High-precision rules for numeric/units patterns (e.g., Ø300mm, 5W30, 125-Zähne)  
2) Noise-robust gazetteers auto-built from the train set and expanded from unlabeled titles  
3) Domain-adapted transformer (`xlm-roberta-base`) fine-tuned for token tagging  
4) Label-wise calibration to optimize Fβ with β = 0.2 (precision > recall)  
5) Self-training (high-confidence pseudo-labels) on the 2M unlabeled titles

The scorer here reproduces the competition metric including weighted Fβ per category and
the final average across the two categories.

## Quickstart

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Drop the provided files into ./data (exact filenames):
#    - Listing_Titles.tsv
#    - Tagged_Titles_Train.tsv

# 3) Train + evaluate on a dev split
python -m ebayner.train --config configs/default.yaml

# 4) Predict on Quiz superset (5,001–30,000 inside Listing_Titles.tsv)
python -m ebayner.predict --config configs/default.yaml --quiz

# 5) Build EvalAI submission (TSV)
python -m ebayner.predict --config configs/default.yaml --quiz --to-submission
# output at outputs/submission.tsv
```

### Repository Layout

```
src/ebayner
  data.py           # strict IO per Annexure, whitespace tokenizer
  rules.py          # regex-based validators + rule-only tagger
  gazetteer.py      # builds/loads noisy gazetteers from train + unlabeled
  model.py          # transformer token tagger (BIO)
  train.py          # training entrypoint + self-training
  predict.py        # inference + merging (rules + gazetteers + model)
  evaluate.py       # official metric reproduction (β=0.2)
  utils.py          # helpers
configs/default.yaml
scripts/
  make_quiz_submission.sh
```

Important: We read TSV via `pandas.read_csv(..., sep="\t", keep_default_na=False, na_values=None)`
to avoid converting blank tags to NA. Writing submissions uses `quoting=csv.QUOTE_NONE`,
as required in the Annexure.

License: MIT.
