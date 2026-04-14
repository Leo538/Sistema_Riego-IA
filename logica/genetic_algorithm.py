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

    if cromosoma[3] > cromosoma[4]:
        penalizacion += (cromosoma[3] - cromosoma[4]) * 6.0

    if cromosoma[4] > cromosoma[6]:
        penalizacion += (cromosoma[4] - cromosoma[6]) * 6.0

    if cromosoma[12] > cromosoma[14]:
        penalizacion += (cromosoma[12] - cromosoma[14]) * 6.0

    if cromosoma[15] > cromosoma[16]:
        penalizacion += (cromosoma[15] - cromosoma[16]) * 6.0

    ancho_h_media = cromosoma[15] - cromosoma[13]
    if ancho_h_media < 12.0:
        penalizacion += (12.0 - ancho_h_media) * 2.0
    if ancho_h_media > 55.0:
        penalizacion += (ancho_h_media - 55.0) * 2.0

    sigma_temp = cromosoma[5]
    if sigma_temp > 9.0:
        penalizacion += (sigma_temp - 9.0) * 4.0

    return penalizacion


def _agua_objetivo(temperatura: float, humedad: float) -> float:
    """
    Referencia física suave para que el AG no premie extremos sin control.
    """
    if humedad <= 25.0:
        base = 6.8
    elif humedad <= 45.0:
        base = 5.2
    elif humedad <= 65.0:
        base = 3.6
    else:
        base = 1.4

    ajuste_temp = float(np.interp(temperatura, [0.0, 25.0, 50.0], [-0.4, 0.0, 0.9]))
    return float(np.clip(base + ajuste_temp, 0.5, 8.2))


def fitness(
    cromosoma: np.ndarray,
    temperatura: float,
    humedad: float,
    tipo_funcion: str,
) -> float:
    """
    Fitness direccional (minimizar).

    - Usa un agua objetivo suave dependiente de humedad y temperatura.
    - Penaliza sobre-riego cuando el suelo ya está húmedo.
    - Evita recompensar el máximo de agua cuando el suelo está muy seco.
    - Suma penalización por formas incoherentes o demasiado extremas.
    """
    try:
        res = calcular_riego(
            temperatura,
            humedad,
            tipo_funcion=tipo_funcion,
            parametros=cromosoma,
        )
        agua = float(res["agua"])
        agua_base = float(
            calcular_riego(
                temperatura,
                humedad,
                tipo_funcion=tipo_funcion,
                parametros=None,
            )["agua"]
        )
    except Exception:
        return 1e9

    error = float(humedad) - HUMEDAD_IDEAL
    incoherencia = _penalizar_incoherencia(cromosoma)
    agua_obj = _agua_objetivo(float(temperatura), float(humedad))
    error_agua = agua - agua_obj
    penalizacion_estabilidad = 0.5 * (agua - agua_base) ** 2

    if error > 0.0:
        penalizacion_agua = ALPHA * (error_agua ** 2) + BETA * max(0.0, agua - 3.0) ** 2
        penalizacion_error = 0.15 * error
    elif error < -20.0:
        penalizacion_agua = ALPHA * (error_agua ** 2) + BETA * max(0.0, 5.5 - agua) ** 2
        penalizacion_error = 0.1 * abs(error)
    else:
        penalizacion_agua = ALPHA * (error_agua ** 2)
        penalizacion_error = BETA * abs(error_agua)

    return penalizacion_error + penalizacion_agua + incoherencia + penalizacion_estabilidad


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


def mutar(c: np.ndarray, sigma_ruido: float = 0.6) -> None:
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
