
from __future__ import annotations
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

def f_beta(prec: float, rec: float, beta: float = 0.2) -> float:
    if prec == 0.0 and rec == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0

def score_category(
    gold: List[Tuple[int, str, str]],
    pred: List[Tuple[int, str, str]],
    category_id: int,
    beta: float = 0.2,
) -> float:
    gold_by_aspect = defaultdict(Counter)
    pred_by_aspect = defaultdict(Counter)

    for rid, name, val in gold:
        gold_by_aspect[name][(rid, val)] += 1
    for rid, name, val in pred:
        pred_by_aspect[name][(rid, val)] += 1

    total_gold = sum(sum(cnt.values()) for cnt in gold_by_aspect.values())

    if total_gold == 0:
        return 0.0

    weighted_sum = 0.0
    for name in set(list(gold_by_aspect.keys()) + list(pred_by_aspect.keys())):
        g = gold_by_aspect[name]
        p = pred_by_aspect[name]
        tp = 0
        for key, gcount in g.items():
            pcount = p.get(key, 0)
            tp += min(gcount, pcount)
        fp = sum(p.values()) - tp
        fn = sum(g.values()) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = f_beta(prec, rec, beta=beta)
        weight = sum(g.values()) / total_gold
        weighted_sum += weight * f
    return weighted_sum

def final_score(
    gold_cat1: List[Tuple[int,str,str]], pred_cat1: List[Tuple[int,str,str]],
    gold_cat2: List[Tuple[int,str,str]], pred_cat2: List[Tuple[int,str,str]],
    beta: float = 0.2
) -> float:
    s1 = score_category(gold_cat1, pred_cat1, category_id=1, beta=beta)
    s2 = score_category(gold_cat2, pred_cat2, category_id=2, beta=beta)
    return (s1 + s2) / 2.0
