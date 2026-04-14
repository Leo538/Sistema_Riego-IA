from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .fitness import fitness
from .operadores import cruce_un_punto, mutar, torneo_binario


def crear_individuo_aleatorio() -> np.ndarray:
    c = np.zeros(18, dtype=float)

    c[0:4] = np.sort(
        np.random.uniform(np.array([0.0, 0.0, 5.0, 15.0]), np.array([0.0, 5.0, 20.0, 30.0]))
    )

    c[4] = float(np.random.uniform(15.0, 35.0))
    c[5] = float(np.random.uniform(3.0, 10.0))

    c[6:9] = np.sort(np.random.uniform(20.0, 50.0, size=3))

    c[9:13] = np.sort(
        np.random.uniform(np.array([0.0, 0.0, 10.0, 25.0]), np.array([0.0, 10.0, 40.0, 60.0]))
    )

    c[13:16] = np.sort(np.random.uniform(20.0, 80.0, size=3))

    c[16:18] = np.sort(np.random.uniform(55.0, 85.0, size=2))

    return c


def ejecutar_genetico(
    temperatura: float,
    humedad: float,
    tipo_funcion: str = "triangular",
    n_generaciones: int = 50,
    tam_poblacion: int = 30,
) -> Dict[str, Any]:
    try:
        n_gen = int(np.clip(n_generaciones, 30, 100))
        n_pop = int(np.clip(tam_poblacion, 20, 50))

        poblacion = [crear_individuo_aleatorio() for _ in range(n_pop)]
        historial: List[float] = []

        fits = np.array([fitness(ind, temperatura, humedad, tipo_funcion) for ind in poblacion])
        mejor_idx = int(np.argmin(fits))
        mejor = poblacion[mejor_idx].copy()
        mejor_fitness = float(fits[mejor_idx])
        historial.append(mejor_fitness)

        for _ in range(n_gen - 1):
            nueva: List[np.ndarray] = []

            nueva.append(mejor.copy())

            while len(nueva) < n_pop:
                pa = torneo_binario(poblacion, fits)
                pb = torneo_binario(poblacion, fits)
                h1, h2 = cruce_un_punto(pa, pb)
                mutar(h1)
                mutar(h2)
                nueva.append(h1)
                if len(nueva) < n_pop:
                    nueva.append(h2)
            poblacion = nueva[:n_pop]
            fits = np.array([fitness(ind, temperatura, humedad, tipo_funcion) for ind in poblacion])
            idx = int(np.argmin(fits))
            if fits[idx] < mejor_fitness:
                mejor_fitness = float(fits[idx])
                mejor = poblacion[idx].copy()
            historial.append(mejor_fitness)

        return {
            "mejor_individuo": mejor.tolist(),
            "mejor_fitness": mejor_fitness,
            "historial": historial,
        }
    except Exception as e:
        raise RuntimeError(f"Error en el algoritmo genético: {e}") from e

