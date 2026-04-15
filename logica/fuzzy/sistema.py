from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .controlador import ENTRADAS_REFERENCIA, calcular_duracion_riego, obtener_curvas_para_graficas
from ..genetico.cromosoma import decodificar_cromosoma


def _parametros_desde_genes(parametros: Optional[np.ndarray]) -> Optional[dict[str, dict[str, tuple[float, ...]]]]:
    if parametros is None:
        return None
    return decodificar_cromosoma(np.asarray(parametros, dtype=float))


def curvas_membresia_para_grafico(parametros: Optional[np.ndarray] = None) -> Dict[str, Any]:
    curvas = obtener_curvas_para_graficas(_parametros_desde_genes(parametros))
    return {
        "temp_x": curvas["temperatura"]["universo"],
        "temp": curvas["temperatura"]["curvas"],
        "humedad_x": curvas["humedad_suelo"]["universo"],
        "humedad": curvas["humedad_suelo"]["curvas"],
        "humedad_rel_x": curvas["humedad_relativa"]["universo"],
        "humedad_rel": curvas["humedad_relativa"]["curvas"],
        "par_x": curvas["par"]["universo"],
        "par": curvas["par"]["curvas"],
        "riego_x": curvas["duracion_riego"]["universo"],
        "riego": {
            "bajo": curvas["duracion_riego"]["curvas"]["corta"],
            "medio": curvas["duracion_riego"]["curvas"]["media"],
            "alto": curvas["duracion_riego"]["curvas"]["larga"],
        },
    }


def calcular_riego(
    temperatura: float,
    humedad: float,
    parametros: Optional[np.ndarray] = None,
    *,
    humedad_relativa: Optional[float] = None,
    par: Optional[float] = None,
) -> Dict[str, Any]:
    hr = float(humedad_relativa) if humedad_relativa is not None else float(ENTRADAS_REFERENCIA["humedad_relativa"])
    par_v = float(par) if par is not None else float(ENTRADAS_REFERENCIA["par"])
    resultado = calcular_duracion_riego(
        humedad_suelo=float(humedad),
        temperatura=float(temperatura),
        humedad_relativa=hr,
        par=par_v,
        parametros_membresia=_parametros_desde_genes(parametros),
    )
    return {
        "agua": resultado["duracion"],
        "duracion": resultado["duracion"],
        "nivel": {"corta": "bajo", "media": "medio", "larga": "alto"}[resultado["nivel"]],
        "nivel_difuso": resultado["nivel"],
        "activaciones": resultado["activaciones"],
        "entradas": resultado["entradas"],
        "configuracion": resultado["configuracion"],
        "grados_entrada": resultado["grados_entrada"],
        "grados_salida": resultado["grados_salida"],
        "centroide": resultado["centroide"],
        "agregado": resultado["agregado"],
        "regla_dominante": resultado["regla_dominante"],
        "top_reglas": resultado["top_reglas"],
        "explicacion": resultado["explicacion"],
    }
