from __future__ import annotations

from typing import Iterable

import numpy as np

from .cromosoma import TOTAL_GENES, reparar_cromosoma


# 2. Selección


def seleccion_torneo_binario(poblacion: list[np.ndarray], fitness_poblacion: np.ndarray) -> np.ndarray:
    i, j = np.random.randint(0, len(poblacion), size=2)
    ganador = i if fitness_poblacion[i] <= fitness_poblacion[j] else j
    return poblacion[ganador].copy()


# 3. Cruce (crossover)


def cruce_un_punto(padre_a: Iterable[float], padre_b: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(list(padre_a), dtype=float)
    b = np.asarray(list(padre_b), dtype=float)
    punto = int(np.random.randint(1, TOTAL_GENES))
    hijo_a = np.concatenate([a[:punto], b[punto:]])
    hijo_b = np.concatenate([b[:punto], a[punto:]])
    return reparar_cromosoma(hijo_a), reparar_cromosoma(hijo_b)


# 4. Mutación


def mutacion_gaussiana(cromosoma: Iterable[float], probabilidad: float = 0.15, sigma: float = 1.25) -> np.ndarray:
    genes = np.asarray(list(cromosoma), dtype=float).copy()
    mascara = np.random.random(size=genes.size) < float(probabilidad)
    ruido = np.random.normal(0.0, float(sigma), size=genes.size)
    genes[mascara] += ruido[mascara]
    return reparar_cromosoma(genes)


def reparar_individuo(cromosoma: Iterable[float]) -> np.ndarray:
    return reparar_cromosoma(cromosoma)
