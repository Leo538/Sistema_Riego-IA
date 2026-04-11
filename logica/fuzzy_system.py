"""
Sistema de inferencia difusa (Mamdani) para recomendación de riego.
Usa scikit-fuzzy; admite parámetros por defecto o un vector de 18 genes (temp + humedad).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import skfuzzy as fuzz


# Universos (resolución adecuada para centroide estable)
_X_TEMP = np.linspace(0, 50, 251)
_X_HUM = np.linspace(0, 100, 501)
_X_RIEGO = np.linspace(0, 10, 501)

_TEMP_MAX = 50.0
_HUM_MAX = 100.0


def _ordenar_trap(a: float, b: float, c: float, d: float) -> List[float]:
    pts = sorted([float(a), float(b), float(c), float(d)])
    return pts


def _ordenar_tri(a: float, b: float, c: float) -> List[float]:
    return sorted([float(a), float(b), float(c)])


def _desempaquetar_genes(parametros: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if parametros is None:
        return None, None
    p = np.asarray(parametros, dtype=float).flatten()
    if p.size != 18:
        raise ValueError("parametros debe tener exactamente 18 valores.")
    return p[0:9], p[9:18]


def _desc_temperatura(tipo_funcion: str, g: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Genes temp (9): [baja a,b,c,d](4), [media centro, sigma](2), [alta a,b,c](3) con d=50.
    """
    if g is None:
        baja = [0.0, 0.0, 10.0, 25.0]
        centro, sigma = 25.0, 8.0
        alta = [25.0, 40.0, 50.0, 50.0]
    else:
        baja = _ordenar_trap(g[0], g[1], g[2], g[3])
        centro = float(np.clip(g[4], 0.0, _TEMP_MAX))
        sigma = max(float(g[5]), 0.5)
        aa, bb, cc = sorted([float(g[6]), float(g[7]), float(g[8])])
        alta = [aa, bb, min(cc, _TEMP_MAX), _TEMP_MAX]

    t = str(tipo_funcion).lower()
    desc: Dict[str, Any] = {"baja_trap": baja, "alta_trap": alta, "media_kind": None, "media_data": None}

    if t == "gaussiana":
        desc["media_kind"] = "gauss"
        desc["media_data"] = (centro, sigma)
    elif t == "triangular":
        desc["media_kind"] = "tri"
        left = max(0.0, centro - 3.0 * sigma)
        right = min(_TEMP_MAX, centro + 3.0 * sigma)
        desc["media_data"] = _ordenar_tri(left, centro, right)
    else:
        desc["media_kind"] = "trap"
        left = max(0.0, centro - 2.0 * sigma)
        mid_l = max(0.0, centro - sigma)
        mid_r = min(_TEMP_MAX, centro + sigma)
        right = min(_TEMP_MAX, centro + 2.0 * sigma)
        desc["media_data"] = _ordenar_trap(left, mid_l, mid_r, right)

    return desc


def _desc_humedad(tipo_funcion: str, g: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Genes hum (9): [baja a,b,c,d](4), [media a,b,c](3), [alta a,b](2) → trap [a,b,100,100].
    En modo gaussiana, la media usa centro en el vértice central del triángulo y sigma
    proporcional a la base (mismos 3 genes, interpretación dual).
    """
    if g is None:
        baja = [0.0, 0.0, 25.0, 50.0]
        media_tri = [20.0, 50.0, 80.0]
        alta = [50.0, 75.0, 100.0, 100.0]
    else:
        baja = _ordenar_trap(g[0], g[1], g[2], g[3])
        media_tri = _ordenar_tri(g[4], g[5], g[6])
        aa, bb = sorted([float(g[7]), float(g[8])])
        alta = [aa, bb, _HUM_MAX, _HUM_MAX]

    a_t, b_t, c_t = media_tri
    gauss_c = float(b_t)
    gauss_s = max((c_t - a_t) / 4.0, 1.0)

    t = str(tipo_funcion).lower()
    return {
        "tipo": t,
        "baja_trap": baja,
        "alta_trap": alta,
        "media_tri": media_tri,
        "media_gauss": (gauss_c, gauss_s),
    }


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
    """Salida fija (no forma parte de los 18 genes)."""
    return {
        "bajo": fuzz.trapmf(x, [0, 0, 2, 5]),
        "medio": fuzz.trimf(x, [2, 5, 8]),
        "alto": fuzz.trapmf(x, [5, 8, 10, 10]),
    }


def curvas_membresia_para_grafico(
    tipo_funcion: str = "triangular",
    parametros: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Datos listos para matplotlib (sin dibujar aquí).
    Retorna ejes y curvas para temperatura, humedad y riego.
    """
    tg, hg = _desempaquetar_genes(parametros)
    dt = _desc_temperatura(tipo_funcion, tg)
    dh = _desc_humedad(tipo_funcion, hg)
    return {
        "temp_x": _X_TEMP,
        "temp": mf_temperatura_arrays(dt, _X_TEMP),
        "humedad_x": _X_HUM,
        "humedad": mf_humedad_arrays(dh, _X_HUM),
        "riego_x": _X_RIEGO,
        "riego": mf_riego_default_arrays(_X_RIEGO),
    }


def _nivel_linguistico(valor: float, x: np.ndarray, mfs: Dict[str, np.ndarray]) -> str:
    idx = int(np.argmin(np.abs(x - valor)))
    etiquetas = ["bajo", "medio", "alto"]
    claves = ["bajo", "medio", "alto"]
    grados = [float(mfs[k][idx]) for k in claves]
    return etiquetas[int(np.argmax(grados))]


def _activaciones_reglas(mu_t: Dict[str, float], mu_h: Dict[str, float]) -> Dict[str, float]:
    """Grado de activación (AND = min) para las 9 reglas Mamdani."""
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


# (humedad_cat, temperatura_cat, riego_cat) — 9 reglas Mamdani
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
    """Agregación max-min y defuzzificación por centroide (universo de riego)."""
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
            return 8.0
        return 4.0
    try:
        return float(fuzz.defuzz(_X_RIEGO, agregado, "centroid"))
    except Exception:
        return float(_X_RIEGO[int(np.argmax(agregado))])


def calcular_riego(
    temperatura: float,
    humedad: float,
    tipo_funcion: str = "triangular",
    parametros: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Inferencia Mamdani + defuzzificación por centroide.

    parametros: 18 floats (temp 9 + humedad 9) o None para valores por defecto.
    """
    try:
        temperatura = float(np.clip(temperatura, 0.0, _TEMP_MAX))
        humedad = float(np.clip(humedad, 0.0, _HUM_MAX))
        tg, hg = _desempaquetar_genes(parametros)
        dt = _desc_temperatura(tipo_funcion, tg)
        dh = _desc_humedad(tipo_funcion, hg)

        t_mf = mf_temperatura_arrays(dt, _X_TEMP)
        h_mf = mf_humedad_arrays(dh, _X_HUM)
        r_mf = mf_riego_default_arrays(_X_RIEGO)

        mu_t = {
            "baja": _membresia_en_punto(_X_TEMP, t_mf["baja"], temperatura),
            "media": _membresia_en_punto(_X_TEMP, t_mf["media"], temperatura),
            "alta": _membresia_en_punto(_X_TEMP, t_mf["alta"], temperatura),
        }
        mu_h = {
            "baja": _membresia_en_punto(_X_HUM, h_mf["baja"], humedad),
            "media": _membresia_en_punto(_X_HUM, h_mf["media"], humedad),
            "alta": _membresia_en_punto(_X_HUM, h_mf["alta"], humedad),
        }
        agua = _inferencia_mamdani_centroide(mu_h, mu_t, r_mf, humedad)
        agua = float(np.clip(agua, 0.0, 10.0))
        activaciones = _activaciones_reglas(mu_t, mu_h)
        nivel = _nivel_linguistico(agua, _X_RIEGO, r_mf)

        return {"agua": agua, "nivel": nivel, "activaciones": activaciones}
    except Exception as e:
        raise RuntimeError(f"Error en el sistema difuso: {e}") from e
