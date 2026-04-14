from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .inferencia import (
    _activaciones_reglas,
    _inferencia_mamdani_centroide,
    _membresia_en_punto,
    _nivel_linguistico,
)
from .membresia import (
    _ordenar_trap,
    _ordenar_tri,
    mf_humedad_arrays,
    mf_riego_default_arrays,
    mf_temperatura_arrays,
)


_X_TEMP = np.linspace(0, 50, 251)
_X_HUM = np.linspace(0, 100, 501)
_X_RIEGO = np.linspace(0, 10, 501)

_TEMP_MAX = 50.0
_HUM_MAX = 100.0


def _desempaquetar_genes(parametros: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if parametros is None:
        return None, None
    p = np.asarray(parametros, dtype=float).flatten()
    if p.size != 18:
        raise ValueError("parametros debe tener exactamente 18 valores.")
    return p[0:9], p[9:18]


def _desc_temperatura(tipo_funcion: str, g: Optional[np.ndarray]) -> Dict[str, Any]:
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


def _formato_num_tabla(x: float) -> str:
    xf = float(x)
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return str(round(xf, 1))


def parametros_originales_temperatura_media_texto(tipo_funcion: str) -> Tuple[str, str]:
    desc = _desc_temperatura(str(tipo_funcion).lower(), None)
    if desc["media_kind"] == "gauss":
        c, s = desc["media_data"]
        return "Gaussiana", f"centro={_formato_num_tabla(c)}, σ={_formato_num_tabla(s)}"
    if desc["media_kind"] == "tri":
        a, b, c = desc["media_data"]
        return (
            "Triangular",
            f"[{_formato_num_tabla(a)}, {_formato_num_tabla(b)}, {_formato_num_tabla(c)}]",
        )
    pts = desc["media_data"]
    return "Trapezoidal", str([_formato_num_tabla(float(p)) for p in pts])


def _desc_humedad(tipo_funcion: str, g: Optional[np.ndarray]) -> Dict[str, Any]:
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
    gauss_s = max((c_t - a_t) / 4.0, 2.0)

    t = str(tipo_funcion).lower()
    return {
        "tipo": t,
        "baja_trap": baja,
        "alta_trap": alta,
        "media_tri": media_tri,
        "media_gauss": (gauss_c, gauss_s),
    }


def curvas_membresia_para_grafico(
    tipo_funcion: str = "triangular",
    parametros: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
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


def calcular_riego(
    temperatura: float,
    humedad: float,
    tipo_funcion: str = "triangular",
    parametros: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
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

