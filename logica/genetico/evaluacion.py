from __future__ import annotations

from typing import Any, Mapping

from ..fuzzy.controlador import calcular_duracion_riego

from .cromosoma import codificar_parametros
from .fitness import evaluar_fitness
from .optimizador import ejecutar_optimizacion


def comparar_sistema_base_y_optimizado(
    entradas: Mapping[str, float],
    generaciones: int = 60,
    tamano_poblacion: int = 40,
) -> dict[str, Any]:
    resultado_base = calcular_duracion_riego(**entradas)
    optimizacion = ejecutar_optimizacion(
        generaciones=generaciones,
        tamano_poblacion=tamano_poblacion,
        entradas_actuales=entradas,
    )
    parametros = optimizacion["parametros_decodificados"]
    resultado_optimizado = calcular_duracion_riego(**entradas, parametros_membresia=parametros)
    fitness_base = evaluar_fitness(codificar_parametros(resultado_base["configuracion"]), entradas_actuales=entradas)
    return {
        "resultado_base": resultado_base,
        "resultado_optimizado": resultado_optimizado,
        "diferencia_duracion": float(resultado_optimizado["duracion"] - resultado_base["duracion"]),
        "mejora_fitness": float(fitness_base - optimizacion["mejor_fitness"]),
        "fitness_base": float(fitness_base),
        "fitness_optimizado": float(optimizacion["mejor_fitness"]),
        "historial": optimizacion["historial"],
        "mejor_individuo": optimizacion["mejor_individuo"],
        "parametros_decodificados": parametros,
    }
