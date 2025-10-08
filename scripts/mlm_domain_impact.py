# scripts/mlm_domain_adapt.py
from __future__ import annotations
import argparse, yaml, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/dapt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    listings = pd.read_csv(cfg["paths"]["listings_tsv"], sep="\t", keep_default_na=False, na_values=None)
    texts = listings["Title"].astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(cfg["training"]["base_model_name"])
    ds = Dataset.from_dict({"text": texts})
    def tok_fn(batch): return tok(batch["text"].split(), is_split_into_words=True, truncation=True, max_length=128)  # whitespace split
    ds = ds.map(lambda ex: tok(ex["text"].split(), is_split_into_words=True, truncation=True, max_length=128), batched=False)
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    model = AutoModelForMaskedLM.from_pretrained(cfg["training"]["base_model_name"])
    args_hf = TrainingArguments(output_dir=args.out, per_device_train_batch_size=64, num_train_epochs=1, learning_rate=5e-5, save_total_limit=1, logging_steps=200)
    Trainer(model=model, args=args_hf, data_collator=collator, train_dataset=ds).train()
    model.save_pretrained(args.out); tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
