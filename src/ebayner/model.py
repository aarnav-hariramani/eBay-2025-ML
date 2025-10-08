
from __future__ import annotations
import os, math, numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments

from .utils import set_seed

from packaging.version import parse as V
import transformers as tf


@dataclass
class NERExample:
    tokens: List[str]
    bio_tags: List[str]
    category_id: int
    record_id: int
    title: str

class NERDataset(Dataset):
    def __init__(self, examples: List[NERExample], tokenizer, label2id: Dict[str,int], max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {v:k for k,v in label2id.items()}
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex.tokens
        labels = ex.bio_tags
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=self.max_length)
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(self.label2id[labels[word_idx]])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        encoding["labels"] = label_ids
        import torch
        return {k: torch.tensor(v) for k,v in encoding.items()}

def create_label_maps(train_labels: List[List[str]]) -> Dict[str,int]:
    uniq = sorted(set([t for seq in train_labels for t in seq]))
    label2id = {lab:i for i,lab in enumerate(uniq)}
    return label2id

def to_examples(seqs: List[dict]) -> List[NERExample]:
    out = []
    for s in seqs:
        out.append(NERExample(tokens=s["tokens"], bio_tags=s["bio_tags"], category_id=s["category_id"], record_id=s["record_id"], title=s["title"]))
    return out

def train_token_classifier(seqs_train: List[dict], seqs_val: List[dict], model_name="xlm-roberta-base", max_length=128, lr=3e-5, batch_size=16, epochs=6, seed=42, work_dir="outputs/model"):
    set_seed(seed)

    lr = float(lr)
    batch_size = int(batch_size)
    epochs = float(epochs)      # HF Trainer supports float epochs (e.g., 2.5)
    max_length = int(max_length)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ex = to_examples(seqs_train)
    val_ex = to_examples(seqs_val)

    label2id = create_label_maps([s["bio_tags"] for s in seqs_train + seqs_val])
    id2label = {v:k for k,v in label2id.items()}

    ds_train = NERDataset(train_ex, tokenizer, label2id, max_length=max_length)
    ds_val = NERDataset(val_ex, tokenizer, label2id, max_length=max_length)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)

    args = TrainingArguments(
    output_dir=work_dir,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    eval_strategy="epoch",      # <-- use eval_strategy on 4.57.0+
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    fp16=False,                 # set True if your GPU supports it
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, args=args, data_collator=data_collator, train_dataset=ds_train, eval_dataset=ds_val)
    trainer.train()
    trainer.save_model(work_dir)
    tokenizer.save_pretrained(work_dir)

    return work_dir, tokenizer, model, label2id

@torch.no_grad()
def predict_tokens(tokens: List[str], model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**encoding).logits[0]  # [seq, labels]
    probs = outputs.softmax(-1).cpu().numpy()
    word_ids = encoding.word_ids()
    preds = []
    for i, wid in enumerate(word_ids):
        if wid is None: 
            continue
        if i > 0 and wid == word_ids[i-1]:
            continue
        label_id = probs[i].argmax()
        label = model.config.id2label[label_id]
        conf = float(probs[i][label_id])
        preds.append((wid, label, conf))
    bio = [ "O" for _ in tokens ]
    confs = [ 0.0 for _ in tokens ]
    for wid, lab, p in preds:
        bio[wid] = lab
        confs[wid] = p
    return bio, confs
