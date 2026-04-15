from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .fuzzy.sistema import calcular_riego
from .genetico.algoritmo import ejecutar_genetico

_ORDEN_NIVEL = {"bajo": 0, "medio": 1, "alto": 2}
_CAIDA_MINIMA_ABSOLUTA_MIN = 3.0
_CAIDA_MINIMA_RELATIVA = 0.20


def degrada_resultado(base: Dict[str, Any], optimizado: Dict[str, Any]) -> bool:
    nivel_base = str(base.get("nivel", "")).lower()
    nivel_opt = str(optimizado.get("nivel", "")).lower()
    orden_base = _ORDEN_NIVEL.get(nivel_base, -1)
    orden_opt = _ORDEN_NIVEL.get(nivel_opt, -1)

    if nivel_base == "alto" and orden_opt >= 0 and orden_opt < orden_base:
        return True

    duracion_base = float(base.get("duracion", base.get("agua", 0.0)))
    duracion_opt = float(optimizado.get("duracion", optimizado.get("agua", 0.0)))
    umbral_caida = max(_CAIDA_MINIMA_ABSOLUTA_MIN, duracion_base * _CAIDA_MINIMA_RELATIVA)
    return (duracion_base - duracion_opt) > umbral_caida


def _motivo_decision(base: Dict[str, Any], optimizado: Dict[str, Any], aceptada: bool) -> str:
    if aceptada:
        return "Se acepta optimización: no se detectó degradación clara frente al caso base."

    nivel_base = str(base.get("nivel", "")).lower()
    nivel_opt = str(optimizado.get("nivel", "")).lower()
    orden_base = _ORDEN_NIVEL.get(nivel_base, -1)
    orden_opt = _ORDEN_NIVEL.get(nivel_opt, -1)
    if nivel_base == "alto" and orden_opt >= 0 and orden_opt < orden_base:
        return (
            "Se rechaza optimización: la etiqueta de salida bajó en un escenario de alta demanda "
            f"({nivel_base} -> {nivel_opt})."
        )

    duracion_base = float(base.get("duracion", base.get("agua", 0.0)))
    duracion_opt = float(optimizado.get("duracion", optimizado.get("agua", 0.0)))
    umbral_caida = max(_CAIDA_MINIMA_ABSOLUTA_MIN, duracion_base * _CAIDA_MINIMA_RELATIVA)
    return (
        "Se rechaza optimización: la duración cayó de forma significativa "
        f"({duracion_base:.2f} -> {duracion_opt:.2f} min; umbral {umbral_caida:.2f} min)."
    )


def generar_datos() -> Dict[str, float]:
    return {
        "humedad_suelo": float(np.random.uniform(0.0, 64.0)),
        "temperatura": float(np.random.uniform(0.0, 50.0)),
        "humedad_relativa": float(np.random.uniform(0.0, 100.0)),
        "par": float(np.random.uniform(0.0, 2000.0)),
    }


def ejecutar_sistema(
    humedad_suelo: float,
    temperatura: float,
    humedad_relativa: float,
    par: float,
) -> Dict[str, Any]:
    return calcular_riego(
        temperatura,
        humedad_suelo,
        parametros=None,
        humedad_relativa=humedad_relativa,
        par=par,
    )


def ejecutar_con_genetico(
    humedad_suelo: float,
    temperatura: float,
    humedad_relativa: float,
    par: float,
    n_generaciones: int = 50,
    tam_poblacion: int = 30,
) -> Dict[str, Any]:
    resultado_sin = ejecutar_sistema(humedad_suelo, temperatura, humedad_relativa, par)
    ga = ejecutar_genetico(
        humedad_suelo,
        temperatura,
        humedad_relativa,
        par,
        n_generaciones=n_generaciones,
        tam_poblacion=tam_poblacion,
    )
    mejor = np.array(ga["mejor_individuo"], dtype=float)
    resultado_opt = calcular_riego(
        temperatura,
        humedad_suelo,
        parametros=mejor,
        humedad_relativa=humedad_relativa,
        par=par,
    )
    se_acepta_optimizacion = not degrada_resultado(resultado_sin, resultado_opt)
    resultado_final = resultado_opt if se_acepta_optimizacion else resultado_sin
    motivo_decision = _motivo_decision(resultado_sin, resultado_opt, se_acepta_optimizacion)

    return {
        "resultado_sin_optimizar": resultado_sin,
        "resultado_optimizado": resultado_opt,
        "resultado_final": resultado_final,
        "se_acepta_optimizacion": se_acepta_optimizacion,
        "motivo_decision": motivo_decision,
        "mejor_fitness": ga["mejor_fitness"],
        "historial": ga["historial"],
        "mejor_individuo": ga["mejor_individuo"],
        "parametros_referencia_operativa": "optimizado" if se_acepta_optimizacion else "base",
    }
