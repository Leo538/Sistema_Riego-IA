from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .optimizador import ejecutar_optimizacion


def ejecutar_genetico(
    humedad_suelo: float,
    temperatura: float,
    humedad_relativa: float,
    par: float,
    n_generaciones: int = 50,
    tam_poblacion: int = 30,
) -> Dict[str, Any]:
    entradas_actuales = {
        "humedad_suelo": float(humedad_suelo),
        "temperatura": float(temperatura),
        "humedad_relativa": float(humedad_relativa),
        "par": float(par),
    }
    resultado = ejecutar_optimizacion(
        generaciones=n_generaciones,
        tamano_poblacion=tam_poblacion,
        entradas_actuales=entradas_actuales,
    )
    return {
        "mejor_individuo": np.asarray(resultado["mejor_individuo"], dtype=float).tolist(),
        "mejor_fitness": resultado["mejor_fitness"],
        "historial": resultado["historial"],
        "parametros_decodificados": resultado["parametros_decodificados"],
    }
