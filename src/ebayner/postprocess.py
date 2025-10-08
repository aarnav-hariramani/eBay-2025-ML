# src/ebayner/postprocess.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict

# ---------- Category applicability from Annexure ----------
# cat 1: Brake kits; cat 2: Timing kits
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

# ---------- High-precision regex validators (precision > recall) ----------
RE_MPN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-/() ]{2,}$")   # general MPN-ish form
RE_OEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-/() ]{1,}$")
RE_DIAM = re.compile(r"(?:^|[^0-9])(?:Ø|O?D\b|Durchmesser|dm|ø)?\s*\d{2,4}\s*mm\b", re.I)
RE_VISC = re.compile(r"^\d{1,2}W\d{2}$", re.I)                # 5W30, 0W20, 10W50
RE_YEAR = re.compile(r"^(?:\d{2}|\d{4})(?:-\d{2,4})?$")
RE_TEETH = re.compile(r"^\d{1,3}(?:-?Z(ä|ae)hne)?$", re.I)
RE_LEN_MM = re.compile(r"^\d{2,5}\s*mm$", re.I)
RE_QTY = re.compile(r"^\d+(?:[\.,]\d+)?\s*(?:L|Gr\.?)$", re.I)
RE_UNIT = re.compile(r"^(mm|zoll|stück|l)$", re.I)

def validate_value(name: str, value: str) -> bool:
    n = name
    v = value.strip()
    if n == "Herstellernummer":          # MPN
        return bool(RE_MPN.match(v)) and any(c.isdigit() for c in v)
    if n == "Oe/Oem_Referenznummer(N)":
        return bool(RE_OEM.match(v)) and any(c.isdigit() for c in v)
    if n == "Bremsscheiben-Aussendurchmesser":
        return bool(RE_DIAM.search(v)) or v.endswith("mm")
    if n == "SAE_Viskosität":
        return bool(RE_VISC.match(v))
    if n == "Kompatibles_Fahrzeug_Jahr":
        return bool(RE_YEAR.match(v))
    if n == "Zähnezahl":
        return bool(RE_TEETH.match(v)) or v.isdigit()
    if n == "Länge":
        return bool(RE_LEN_MM.match(v)) or v.isdigit()
    if n == "Menge":
        return bool(RE_QTY.match(v))
    if n == "Maßeinheit":
        return bool(RE_UNIT.match(v))
    # default: allow
    return True

def join_tokens(tokens: List[str]) -> str:
    # competition requires tokens joined with single ASCII space 0x20. :contentReference[oaicite:2]{index=2}
    return " ".join(tokens).strip()

def filter_and_gate(
    cat_id: int,
    spans: List[Tuple[str, List[str], float]],
    thresholds: Dict[str, float] | None = None,
) -> List[Tuple[str, str]]:
    """
    Args:
      spans: list of (aspect_name, token_span, conf) from the BIO decoder
      thresholds: per-aspect min confidence; default 0.80 unless provided
    Returns:
      list of (aspect_name, value_str) kept for submission
    """
    kept = []
    ok_set = CAT_OK.get(int(cat_id), set())
    for name, toks, conf in spans:
        if name not in ok_set:                # gate by category applicability
            continue
        thr = (thresholds or {}).get(name, 0.80)
        if conf < thr:
            continue
        value = join_tokens(toks)
        if not validate_value(name, value):
            continue
        kept.append((name, value))
    return kept
