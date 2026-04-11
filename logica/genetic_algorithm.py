"""
Algoritmo genético para optimizar los parámetros de membresía (18 genes)
del sistema difuso de riego.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .fuzzy_system import calcular_riego

# Pesos de la función de fitness (minimizar)
ALPHA = 0.7
BETA = 0.3
HUMEDAD_IDEAL = 60.0

PROB_MUTACION = 0.2


def _reparar_trap_segment(c: np.ndarray, i0: int, u_max: float) -> None:
    seg = np.clip(c[i0 : i0 + 4], 0.0, u_max)
    seg.sort()
    c[i0 : i0 + 4] = seg


def _reparar_temp_alta(c: np.ndarray) -> None:
    seg = np.clip(c[6:9], 20.0, 50.0)
    seg.sort()
    c[6:9] = seg


def _reparar_hum_alta(c: np.ndarray) -> None:
    seg = np.clip(c[16:18], 55.0, 85.0)
    seg.sort()
    c[16:18] = seg


def crear_individuo_aleatorio() -> np.ndarray:
    """Genera un cromosoma válido (18 genes) con rangos lógicos entre MF."""
    c = np.zeros(18, dtype=float)

    # Temperatura baja: empieza en 0 y termina antes de 30
    c[0:4] = np.sort(
        np.random.uniform(np.array([0.0, 0.0, 5.0, 15.0]), np.array([0.0, 5.0, 20.0, 30.0]))
    )

    # Temperatura media: centro entre 15 y 35
    c[4] = float(np.random.uniform(15.0, 35.0))
    c[5] = float(np.random.uniform(3.0, 10.0))

    # Temperatura alta: después de 20 hasta 50
    c[6:9] = np.sort(np.random.uniform(20.0, 50.0, size=3))

    # Humedad baja: empieza en 0 y termina antes de 60
    c[9:13] = np.sort(
        np.random.uniform(np.array([0.0, 0.0, 10.0, 25.0]), np.array([0.0, 10.0, 40.0, 60.0]))
    )

    # Humedad media: triángulo entre 20 y 80
    c[13:16] = np.sort(np.random.uniform(20.0, 80.0, size=3))

    # Humedad alta: después de 40 (genes hasta 100 en el sistema difuso)
    c[16:18] = np.sort(np.random.uniform(55.0, 85.0, size=2))

    return c


def reparar_cromosoma(c: np.ndarray) -> None:
    """In-place: orden y rangos coherentes."""
    _reparar_trap_segment(c, 0, 30.0)
    c[4] = float(np.clip(c[4], 15.0, 35.0))
    c[5] = float(np.clip(c[5], 3.0, 10.0))
    _reparar_temp_alta(c)
    _reparar_trap_segment(c, 9, 60.0)
    seg = np.clip(c[13:16], 20.0, 80.0)
    seg.sort()
    c[13:16] = seg
    _reparar_hum_alta(c)


def _penalizar_incoherencia(cromosoma: np.ndarray) -> float:
    """
    Penaliza solapes ilógicos: el fin de "baja" no debe quedar después del inicio de "alta".
    """
    penalizacion = 0.0

    if cromosoma[3] > cromosoma[6]:
        penalizacion += (cromosoma[3] - cromosoma[6]) * 10.0

    if cromosoma[12] > cromosoma[16]:
        penalizacion += (cromosoma[12] - cromosoma[16]) * 10.0

    return penalizacion


def fitness(
    cromosoma: np.ndarray,
    temperatura: float,
    humedad: float,
    tipo_funcion: str,
) -> float:
    """
    Fitness direccional (minimizar).

    - Suelo demasiado húmedo (h > ideal): penaliza fuerte el riego.
    - Suelo muy seco (h < ideal − 20): error con α y penalización suave al agua (0,5·β·agua).
    - Resto: error con α y agua con β.
    - Suma penalización por MF incoherentes (solape baja/alta).
    """
    try:
        res = calcular_riego(
            temperatura,
            humedad,
            tipo_funcion=tipo_funcion,
            parametros=cromosoma,
        )
        agua = float(res["agua"])
    except Exception:
        return 1e9

    error = float(humedad) - HUMEDAD_IDEAL
    incoherencia = _penalizar_incoherencia(cromosoma)

    if error > 0:
        penalizacion_agua = agua * 2.0
        penalizacion_error = ALPHA * error
    elif error < -20.0:
        # Suelo muy seco pero penalizar algo el agua para evitar favorecer el máximo sin límite
        penalizacion_agua = BETA * 0.5 * agua
        penalizacion_error = ALPHA * abs(error)
    else:
        penalizacion_agua = BETA * agua
        penalizacion_error = ALPHA * abs(error)

    return penalizacion_error + penalizacion_agua + incoherencia


def torneo_binario(poblacion: List[np.ndarray], fits: np.ndarray) -> np.ndarray:
    i, j = np.random.randint(0, len(poblacion), size=2)
    return poblacion[i].copy() if fits[i] < fits[j] else poblacion[j].copy()


def cruce_un_punto(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    punto = int(np.random.randint(1, 18))
    h1 = p1.copy()
    h2 = p2.copy()
    h1[punto:] = p2[punto:]
    h2[punto:] = p1[punto:]
    reparar_cromosoma(h1)
    reparar_cromosoma(h2)
    return h1, h2


def mutar(c: np.ndarray, sigma_ruido: float = 1.0) -> None:
    for i in range(18):
        if np.random.random() < PROB_MUTACION:
            c[i] += float(np.random.normal(0.0, sigma_ruido))
    reparar_cromosoma(c)


def ejecutar_genetico(
    temperatura: float,
    humedad: float,
    tipo_funcion: str = "triangular",
    n_generaciones: int = 50,
    tam_poblacion: int = 30,
) -> Dict[str, Any]:
    """
    Evoluciona la población y devuelve el mejor individuo y el historial de fitness.
    """
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

            # Elitismo: el mejor individuo pasa intacto a la siguiente generación
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
