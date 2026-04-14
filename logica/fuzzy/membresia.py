from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import skfuzzy as fuzz


_TEMP_MAX = 50.0
_HUM_MAX = 100.0


def _ordenar_trap(a: float, b: float, c: float, d: float) -> List[float]:
    pts = sorted([float(a), float(b), float(c), float(d)])
    return pts


def _ordenar_tri(a: float, b: float, c: float) -> List[float]:
    return sorted([float(a), float(b), float(c)])


def mf_temperatura_arrays(desc: Dict[str, Any], x: np.ndarray) -> Dict[str, np.ndarray]:
    yb = fuzz.trapmf(x, desc["baja_trap"])
    ya = fuzz.trapmf(x, desc["alta_trap"])
    if desc["media_kind"] == "gauss":
        c, s = desc["media_data"]
        ym = fuzz.gaussmf(x, c, s)
    elif desc["media_kind"] == "tri":
        ym = fuzz.trimf(x, desc["media_data"])
    else:
        ym = fuzz.trapmf(x, desc["media_data"])
    return {"baja": yb, "media": ym, "alta": ya}


def mf_humedad_arrays(dh: Dict[str, Any], x: np.ndarray) -> Dict[str, np.ndarray]:
    yb = fuzz.trapmf(x, dh["baja_trap"])
    ya = fuzz.trapmf(x, dh["alta_trap"])
    if dh["tipo"] == "gaussiana":
        c, s = dh["media_gauss"]
        ym = fuzz.gaussmf(x, c, s)
    elif dh["tipo"] == "triangular":
        ym = fuzz.trimf(x, dh["media_tri"])
    else:
        c, s = dh["media_gauss"]
        left = max(0.0, c - 2.0 * s)
        mid_l = max(0.0, c - s)
        mid_r = min(_HUM_MAX, c + s)
        right = min(_HUM_MAX, c + 2.0 * s)
        ym = fuzz.trapmf(x, _ordenar_trap(left, mid_l, mid_r, right))
    return {"baja": yb, "media": ym, "alta": ya}


def mf_riego_default_arrays(x: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "bajo": fuzz.trapmf(x, [0, 0, 2, 5]),
        "medio": fuzz.trimf(x, [2, 5, 8]),
        "alto": fuzz.trapmf(x, [5, 8, 10, 10]),
    }

