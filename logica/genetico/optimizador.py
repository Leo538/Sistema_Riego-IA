from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .cromosoma import crear_individuo_aleatorio, decodificar_cromosoma
from .fitness import evaluar_fitness
from .operadores import cruce_un_punto, mutacion_gaussiana, seleccion_torneo_binario


# 6. Bucle evolutivo (generaciones, elitismo)


def ejecutar_optimizacion(
    generaciones: int = 60,
    tamano_poblacion: int = 40,
    elitismo: int = 2,
    probabilidad_mutacion: float = 0.15,
    sigma_mutacion: float = 1.25,
    entradas_actuales: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    total_generaciones = max(10, int(generaciones))
    total_poblacion = max(6, int(tamano_poblacion))
    cantidad_elites = max(1, min(int(elitismo), total_poblacion))

    poblacion = [crear_individuo_aleatorio() for _ in range(total_poblacion)]
    historial: list[float] = []

    fitness_actual = np.array(
        [evaluar_fitness(individuo, entradas_actuales=entradas_actuales) for individuo in poblacion],
        dtype=float,
    )
    mejor_indice = int(np.argmin(fitness_actual))
    mejor_individuo = poblacion[mejor_indice].copy()
    mejor_fitness = float(fitness_actual[mejor_indice])

    for _ in range(total_generaciones):
        orden = np.argsort(fitness_actual)
        elites = [poblacion[int(indice)].copy() for indice in orden[:cantidad_elites]]
        nueva_poblacion: list[np.ndarray] = elites.copy()

        while len(nueva_poblacion) < total_poblacion:
            padre_a = seleccion_torneo_binario(poblacion, fitness_actual)
            padre_b = seleccion_torneo_binario(poblacion, fitness_actual)
            hijo_a, hijo_b = cruce_un_punto(padre_a, padre_b)
            hijo_a = mutacion_gaussiana(hijo_a, probabilidad=probabilidad_mutacion, sigma=sigma_mutacion)
            hijo_b = mutacion_gaussiana(hijo_b, probabilidad=probabilidad_mutacion, sigma=sigma_mutacion)
            nueva_poblacion.append(hijo_a)
            if len(nueva_poblacion) < total_poblacion:
                nueva_poblacion.append(hijo_b)

        poblacion = nueva_poblacion[:total_poblacion]
        fitness_actual = np.array(
            [evaluar_fitness(individuo, entradas_actuales=entradas_actuales) for individuo in poblacion],
            dtype=float,
        )
        indice_generacion = int(np.argmin(fitness_actual))
        if float(fitness_actual[indice_generacion]) < mejor_fitness:
            mejor_fitness = float(fitness_actual[indice_generacion])
            mejor_individuo = poblacion[indice_generacion].copy()
        historial.append(mejor_fitness)

    parametros = decodificar_cromosoma(mejor_individuo)
    return {
        "mejor_individuo": mejor_individuo,
        "mejor_fitness": mejor_fitness,
        "historial": historial,
        "parametros_decodificados": parametros,
    }
