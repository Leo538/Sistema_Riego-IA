from __future__ import annotations

from typing import Iterable

import numpy as np

from ..fuzzy.membresia import ORDEN_ENTRADAS, obtener_parametros_por_defecto


TOTAL_GENES = 44

RANGOS_VARIABLES = {
    "humedad_suelo": (0.0, 64.0),
    "temperatura": (0.0, 50.0),
    "humedad_relativa": (0.0, 100.0),
    "par": (0.0, 2000.0),
}


# 1. Población — codificación, reparación y creación de individuos


def _segmentos_variable(valores: Iterable[float]) -> dict[str, tuple[float, ...]]:
    datos = [float(v) for v in valores]
    if len(datos) != 11:
        raise ValueError("Cada variable debe tener exactamente 11 parámetros.")
    return {
        "baja": tuple(datos[0:4]),
        "media": tuple(datos[4:7]),
        "alta": tuple(datos[7:11]),
    }


def reparar_segmento_variable(variable: str, segmento: Iterable[float]) -> np.ndarray:
    minimo, maximo = RANGOS_VARIABLES[variable]
    datos = np.clip(np.asarray(list(segmento), dtype=float), minimo, maximo)
    datos[0:4] = np.sort(datos[0:4])
    datos[4:7] = np.sort(datos[4:7])
    datos[7:11] = np.sort(datos[7:11])

    datos[4] = max(datos[4], datos[1])
    datos[5] = min(max(datos[5], datos[4]), datos[6])
    datos[6] = max(datos[6], datos[2])

    if datos[3] > datos[7]:
        frontera = (datos[3] + datos[7]) / 2.0
        datos[3] = frontera
        datos[7] = frontera

    return np.clip(datos, minimo, maximo)


def reparar_cromosoma(cromosoma: Iterable[float]) -> np.ndarray:
    genes = np.asarray(list(cromosoma), dtype=float).flatten()
    if genes.size != TOTAL_GENES:
        raise ValueError(f"El cromosoma debe tener {TOTAL_GENES} genes.")

    reparado = genes.copy()
    for indice, variable in enumerate(ORDEN_ENTRADAS):
        inicio = indice * 11
        fin = inicio + 11
        reparado[inicio:fin] = reparar_segmento_variable(variable, reparado[inicio:fin])
    return reparado


def codificar_parametros(parametros: dict[str, dict[str, tuple[float, ...]]]) -> np.ndarray:
    genes: list[float] = []
    for variable in ORDEN_ENTRADAS:
        definicion = parametros[variable]
        genes.extend(definicion["baja"])
        genes.extend(definicion["media"])
        genes.extend(definicion["alta"])
    return reparar_cromosoma(genes)


def decodificar_cromosoma(cromosoma: Iterable[float]) -> dict[str, dict[str, tuple[float, ...]]]:
    reparado = reparar_cromosoma(cromosoma)
    parametros = obtener_parametros_por_defecto()
    for indice, variable in enumerate(ORDEN_ENTRADAS):
        inicio = indice * 11
        fin = inicio + 11
        parametros[variable] = _segmentos_variable(reparado[inicio:fin])
    return parametros


def crear_individuo_aleatorio() -> np.ndarray:
    genes: list[float] = []
    for variable in ORDEN_ENTRADAS:
        minimo, maximo = RANGOS_VARIABLES[variable]
        amplitud = maximo - minimo
        baja = np.sort(
            np.random.uniform(
                [minimo, minimo, minimo + amplitud * 0.08, minimo + amplitud * 0.2],
                [minimo, minimo + amplitud * 0.08, minimo + amplitud * 0.3, minimo + amplitud * 0.45],
            )
        )
        media = np.sort(
            np.random.uniform(
                [minimo + amplitud * 0.25, minimo + amplitud * 0.4, minimo + amplitud * 0.55],
                [minimo + amplitud * 0.45, minimo + amplitud * 0.6, minimo + amplitud * 0.8],
            )
        )
        alta = np.sort(
            np.random.uniform(
                [minimo + amplitud * 0.55, minimo + amplitud * 0.72, maximo, maximo],
                [minimo + amplitud * 0.75, minimo + amplitud * 0.92, maximo, maximo],
            )
        )
        genes.extend(baja.tolist())
        genes.extend(media.tolist())
        genes.extend(alta.tolist())
    return reparar_cromosoma(genes)
