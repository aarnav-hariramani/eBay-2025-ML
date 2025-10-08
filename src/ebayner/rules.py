
from __future__ import annotations
import re
from typing import List, Tuple, Dict

# Regex validators
RE_MM = re.compile(r"(?i)\b\d+(\.\d+)?\s*mm\b|Ø\s*\d+(\.\d+)?\s*mm|Ø\d+")
RE_YEAR = re.compile(r"\b(19|20)\d{2}\b|\b\d{2}-\d{2}\b")
RE_SAE = re.compile(r"\b\d{1,2}W\d{2}\b")
RE_MPN = re.compile(r"[A-Z0-9][A-Z0-9\-/\.]{3,}")
RE_QTY = re.compile(r"\b(\d+|[1-9]x|[1-9]\s*x|[1-9]-teilig)\b", re.IGNORECASE)
RE_TEETH = re.compile(r"\b\d{2,3}(-|\s*)Z(ä|ae|a)hne\b|\b\d{2,3}\b")  # heuristic

# Common German connectors we ignore in rules
CONNECTORS = {"für","mit","und","oder","der","die","das","passend","+"}

# High-precision rule tagger (returns list of (name, value) pairs)
def rule_tag(tokens: List[str]) -> List[Tuple[str,str]]:
    ents = []
    # Simple scanning windows
    for i,t in enumerate(tokens):
        low = t.lower()
        if low in CONNECTORS:
            continue
        # SAE viscosity
        if RE_SAE.fullmatch(t):
            ents.append(("SAE_Viskosität", t))
            continue
        # Größe / Maßeinheit / Länge / Breite heuristics
        if RE_MM.search(t) or t.endswith("mm"):
            ents.append(("Maßeinheit", "mm"))
        # possible part numbers & OE: favor if has mix of letters+digits and length >= 5
        if RE_MPN.fullmatch(t) and any(c.isalpha() for c in t) and any(c.isdigit() for c in t):
            ents.append(("Herstellernummer", t))
    return ents

# Validators per tag to boost precision
def validate(tag: str, value: str) -> bool:
    if tag == "SAE_Viskosität":
        return bool(RE_SAE.fullmatch(value))
    if tag in ("Maßeinheit","Größe","Länge","Breite"):
        return "mm" in value or "MM" in value or "Ø" in value
    if tag in ("Herstellernummer","Oe/Oem_Referenznummer(N)"):
        return bool(RE_MPN.fullmatch(value)) and any(c.isalpha() for c in value) and any(c.isdigit() for c in value)
    if tag == "Kompatibles_Fahrzeug_Jahr":
        return bool(RE_YEAR.search(value))
    if tag == "Zähnezahl":
        return any(ch.isdigit() for ch in value)
    return True
