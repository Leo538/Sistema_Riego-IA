"""
Capa de coordinación entre el sistema difuso y el algoritmo genético.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .fuzzy_system import calcular_riego
from .genetic_algorithm import ejecutar_genetico


def generar_datos() -> Dict[str, float]:
    """Temperatura y humedad aleatorias en sus rangos físicos."""
    return {
        "temperatura": float(np.random.uniform(0.0, 50.0)),
        "humedad": float(np.random.uniform(0.0, 100.0)),
    }


def ejecutar_sistema(temperatura: float, humedad: float, tipo_funcion: str = "triangular") -> Dict[str, Any]:
    """Ejecuta solo el sistema difuso (parámetros por defecto)."""
    return calcular_riego(temperatura, humedad, tipo_funcion=tipo_funcion, parametros=None)


def ejecutar_con_genetico(
    temperatura: float,
    humedad: float,
    tipo_funcion: str = "triangular",
    n_generaciones: int = 50,
    tam_poblacion: int = 30,
) -> Dict[str, Any]:
    """
    Optimiza membresías con AG y compara salida optimizada vs. por defecto.
    """
    resultado_sin = ejecutar_sistema(temperatura, humedad, tipo_funcion=tipo_funcion)
    ga = ejecutar_genetico(
        temperatura,
        humedad,
        tipo_funcion=tipo_funcion,
        n_generaciones=n_generaciones,
        tam_poblacion=tam_poblacion,
    )
    mejor = np.array(ga["mejor_individuo"], dtype=float)
    resultado_opt = calcular_riego(temperatura, humedad, tipo_funcion=tipo_funcion, parametros=mejor)
    return {
        "resultado_sin_optimizar": resultado_sin,
        "resultado_optimizado": resultado_opt,
        "mejor_fitness": ga["mejor_fitness"],
        "historial": ga["historial"],
        "mejor_individuo": ga["mejor_individuo"],
    }
