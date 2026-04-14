from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import skfuzzy as fuzz


_X_RIEGO = np.linspace(0, 10, 501)


def _nivel_linguistico(valor: float, x: np.ndarray, mfs: Dict[str, np.ndarray]) -> str:
    idx = int(np.argmin(np.abs(x - valor)))
    etiquetas = ["bajo", "medio", "alto"]
    claves = ["bajo", "medio", "alto"]
    grados = [float(mfs[k][idx]) for k in claves]
    return etiquetas[int(np.argmax(grados))]


def _activaciones_reglas(mu_t: Dict[str, float], mu_h: Dict[str, float]) -> Dict[str, float]:
    return {
        "R1_humedad_baja_temp_alta_riego_alto": min(mu_h["baja"], mu_t["alta"]),
        "R2_humedad_baja_temp_media_riego_alto": min(mu_h["baja"], mu_t["media"]),
        "R3_humedad_baja_temp_baja_riego_medio": min(mu_h["baja"], mu_t["baja"]),
        "R4_humedad_media_temp_alta_riego_medio": min(mu_h["media"], mu_t["alta"]),
        "R5_humedad_media_temp_media_riego_medio": min(mu_h["media"], mu_t["media"]),
        "R6_humedad_media_temp_baja_riego_bajo": min(mu_h["media"], mu_t["baja"]),
        "R7_humedad_alta_temp_alta_riego_bajo": min(mu_h["alta"], mu_t["alta"]),
        "R8_humedad_alta_temp_media_riego_bajo": min(mu_h["alta"], mu_t["media"]),
        "R9_humedad_alta_temp_baja_riego_bajo": min(mu_h["alta"], mu_t["baja"]),
    }


def _membresia_en_punto(x_arr: np.ndarray, y_arr: np.ndarray, x0: float) -> float:
    return float(fuzz.interp_membership(x_arr, y_arr, x0))


_REGLAS_MAMDANI: List[Tuple[str, str, str]] = [
    ("baja", "alta", "alto"),
    ("baja", "media", "alto"),
    ("baja", "baja", "medio"),
    ("media", "alta", "medio"),
    ("media", "media", "medio"),
    ("media", "baja", "bajo"),
    ("alta", "alta", "bajo"),
    ("alta", "media", "bajo"),
    ("alta", "baja", "bajo"),
]


def _inferencia_mamdani_centroide(
    mu_h: Dict[str, float],
    mu_t: Dict[str, float],
    r_mf: Dict[str, np.ndarray],
    humedad: float,
) -> float:
    agregado = np.zeros_like(_X_RIEGO, dtype=float)
    for kh, kt, ko in _REGLAS_MAMDANI:
        act = min(mu_h[kh], mu_t[kt])
        if act <= 0.0:
            continue
        recortada = np.minimum(r_mf[ko], act)
        agregado = np.maximum(agregado, recortada)
    if float(np.max(agregado)) < 1e-9:
        if humedad > 70.0:
            return 1.0
        if humedad < 30.0:
            return 7.5
        return 4.0
    try:
        return float(fuzz.defuzz(_X_RIEGO, agregado, "centroid"))
    except Exception:
        return float(_X_RIEGO[int(np.argmax(agregado))])

