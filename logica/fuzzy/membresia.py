from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class DefinicionVariable:
    nombre: str
    minimo: float
    maximo: float
    puntos: int
    etiquetas: Mapping[str, tuple[float, ...]]


VARIABLES_DIFUSAS: dict[str, DefinicionVariable] = {
    "humedad_suelo": DefinicionVariable(
        nombre="humedad_suelo",
        minimo=0.0,
        maximo=64.0,
        puntos=641,
        etiquetas={
            "baja": (0.0, 0.0, 15.0, 28.0),
            "media": (22.0, 36.0, 50.0),
            "alta": (44.0, 56.0, 64.0, 64.0),
        },
    ),
    "temperatura": DefinicionVariable(
        nombre="temperatura",
        minimo=0.0,
        maximo=50.0,
        puntos=501,
        etiquetas={
            "baja": (0.0, 0.0, 12.0, 20.0),
            "media": (18.0, 25.0, 32.0),
            "alta": (30.0, 38.0, 50.0, 50.0),
        },
    ),
    "humedad_relativa": DefinicionVariable(
        nombre="humedad_relativa",
        minimo=0.0,
        maximo=100.0,
        puntos=1001,
        etiquetas={
            "baja": (0.0, 0.0, 30.0, 45.0),
            "media": (40.0, 60.0, 80.0),
            "alta": (70.0, 85.0, 100.0, 100.0),
        },
    ),
    "par": DefinicionVariable(
        nombre="par",
        minimo=0.0,
        maximo=2000.0,
        puntos=1001,
        etiquetas={
            "baja": (0.0, 0.0, 300.0, 700.0),
            "media": (500.0, 1000.0, 1500.0),
            "alta": (1300.0, 1700.0, 2000.0, 2000.0),
        },
    ),
    "duracion_riego": DefinicionVariable(
        nombre="duracion_riego",
        minimo=0.0,
        maximo=30.0,
        puntos=601,
        etiquetas={
            "corta": (0.0, 0.0, 4.0, 10.0),
            "media": (8.0, 15.0, 22.0),
            "larga": (20.0, 26.0, 30.0, 30.0),
        },
    ),
}

ORDEN_ENTRADAS = ("humedad_suelo", "temperatura", "humedad_relativa", "par")


# 1. Fuzzificación — universos y funciones de membresía


def construir_universo(minimo: float, maximo: float, puntos: int) -> np.ndarray:
    return np.linspace(float(minimo), float(maximo), int(puntos), dtype=float)


def trimf(universo: np.ndarray, parametros: Iterable[float]) -> np.ndarray:
    a, b, c = [float(v) for v in parametros]
    x = np.asarray(universo, dtype=float)
    if not (a <= b <= c):
        raise ValueError("Los parámetros triangulares deben cumplir a <= b <= c.")

    y = np.zeros_like(x, dtype=float)
    if b > a:
        mascara_izquierda = (a < x) & (x < b)
        y[mascara_izquierda] = (x[mascara_izquierda] - a) / (b - a)
    y[x == b] = 1.0
    if c > b:
        mascara_derecha = (b < x) & (x < c)
        y[mascara_derecha] = (c - x[mascara_derecha]) / (c - b)
    if a == b:
        y[x == a] = 1.0
    if b == c:
        y[x == c] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(universo: np.ndarray, parametros: Iterable[float]) -> np.ndarray:
    a, b, c, d = [float(v) for v in parametros]
    x = np.asarray(universo, dtype=float)
    if not (a <= b <= c <= d):
        raise ValueError("Los parámetros trapezoidales deben cumplir a <= b <= c <= d.")

    y = np.zeros_like(x, dtype=float)
    if b > a:
        mascara_subida = (a < x) & (x < b)
        y[mascara_subida] = (x[mascara_subida] - a) / (b - a)
    y[(b <= x) & (x <= c)] = 1.0
    if d > c:
        mascara_bajada = (c < x) & (x < d)
        y[mascara_bajada] = (d - x[mascara_bajada]) / (d - c)
    if a == b:
        y[x == a] = 1.0
    if c == d:
        y[x == d] = 1.0
    return np.clip(y, 0.0, 1.0)


def interpolar_membresia(universo: np.ndarray, curva: np.ndarray, valor: float) -> float:
    return float(np.interp(float(valor), np.asarray(universo, dtype=float), np.asarray(curva, dtype=float)))


def obtener_parametros_por_defecto() -> dict[str, dict[str, tuple[float, ...]]]:
    return {
        nombre: {etiqueta: tuple(parametros) for etiqueta, parametros in variable.etiquetas.items()}
        for nombre, variable in VARIABLES_DIFUSAS.items()
    }


def _normalizar_parametros(
    parametros_entrada: Mapping[str, Mapping[str, Iterable[float]]] | None,
) -> dict[str, dict[str, tuple[float, ...]]]:
    parametros = obtener_parametros_por_defecto()
    if parametros_entrada is None:
        return parametros

    for variable, etiquetas in parametros_entrada.items():
        if variable not in parametros:
            continue
        for etiqueta, valores in etiquetas.items():
            if etiqueta in parametros[variable]:
                parametros[variable][etiqueta] = tuple(float(v) for v in valores)
    return parametros


def construir_membresias(
    parametros: Mapping[str, Mapping[str, Iterable[float]]] | None = None,
) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
    configuracion = _normalizar_parametros(parametros)
    resultado: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = {}
    for nombre_variable, definicion in VARIABLES_DIFUSAS.items():
        universo = construir_universo(definicion.minimo, definicion.maximo, definicion.puntos)
        curvas: dict[str, np.ndarray] = {}
        for etiqueta, parametros_etiqueta in configuracion[nombre_variable].items():
            if len(parametros_etiqueta) == 3:
                curvas[etiqueta] = trimf(universo, parametros_etiqueta)
            elif len(parametros_etiqueta) == 4:
                curvas[etiqueta] = trapmf(universo, parametros_etiqueta)
            else:
                raise ValueError(
                    f"La etiqueta '{etiqueta}' de '{nombre_variable}' debe tener 3 o 4 parámetros."
                )
        resultado[nombre_variable] = {"universo": universo, "curvas": curvas}
    return resultado


def construir_membresias_desde_parametros_optimizados(
    parametros_entrada: Mapping[str, Mapping[str, Iterable[float]]],
) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
    return construir_membresias(parametros_entrada)


def describir_configuracion(
    parametros: Mapping[str, Mapping[str, Iterable[float]]] | None = None,
) -> dict[str, dict[str, tuple[float, ...]]]:
    return _normalizar_parametros(parametros)
