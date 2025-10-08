
from __future__ import annotations
import os, json, re
from collections import Counter, defaultdict
from typing import Dict, Set, List, Tuple
from .utils import ensure_dir

def build_gazetteers_from_train(seqs: List[dict], save_dir: str) -> Dict[str, Set[str]]:
    ensure_dir(save_dir)
    gaz = defaultdict(set)
    for s in seqs:
        toks, bio = s["tokens"], s["bio_tags"]
        i = 0
        while i < len(toks):
            tag = bio[i]
            if tag.startswith("B-"):
                name = tag[2:]
                val = [toks[i]]
                j = i+1
                while j < len(toks) and bio[j] == f"I-{name}":
                    val.append(toks[j])
                    j += 1
                value = " ".join(val)
                gaz[name].add(value)
                i = j
            else:
                i += 1
    path = os.path.join(save_dir, "gazetteers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: sorted(list(v)) for k,v in gaz.items()}, f, ensure_ascii=False, indent=2)
    return gaz

def load_gazetteers(save_dir: str) -> Dict[str, Set[str]]:
    path = os.path.join(save_dir, "gazetteers.json")
    if not os.path.exists(path):
        return {}
    import json
    data = json.load(open(path, "r", encoding="utf-8"))
    return {k: set(v) for k,v in data.items()}

def match_gazetteers(tokens: List[str], gaz: Dict[str, Set[str]]) -> List[Tuple[str,str]]:
    outs = []
    max_len = 4
    for name, entries in gaz.items():
        for n in range(1, max_len+1):
            for i in range(0, len(tokens)-n+1):
                phrase = " ".join(tokens[i:i+n])
                if phrase in entries:
                    outs.append((name, phrase))
    return outs
