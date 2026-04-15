from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .membresia import interpolar_membresia
from .reglas import FuzzyRule, obtener_reglas


# 1. Fuzzificación — grados de pertenencia de las entradas


def calcular_membresias_entrada(
    entradas: Mapping[str, float],
    definicion_membresias: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    grados: dict[str, dict[str, float]] = {}
    for variable, valor in entradas.items():
        universo = np.asarray(definicion_membresias[variable]["universo"], dtype=float)
        curvas = definicion_membresias[variable]["curvas"]
        grados[variable] = {
            etiqueta: interpolar_membresia(universo, np.asarray(curva, dtype=float), float(valor))
            for etiqueta, curva in curvas.items()
        }
    return grados


# 2. Inferencia — activación de reglas (Mamdani, AND = mínimo)


def activar_reglas(
    grados_entrada: Mapping[str, Mapping[str, float]],
    reglas: list[FuzzyRule] | None = None,
) -> dict[str, dict[str, Any]]:
    reglas_activas = reglas or obtener_reglas()
    activaciones: dict[str, dict[str, Any]] = {}
    for regla in reglas_activas:
        grado = min(grados_entrada[variable][etiqueta] for variable, etiqueta in regla.antecedents)
        activaciones[regla.name] = {
            "grado": float(grado) * float(regla.weight),
            "salida": regla.consequent[1],
            "variable_salida": regla.consequent[0],
            "antecedentes": dict(regla.antecedents),
            "descripcion": regla.description,
            "peso": regla.weight,
        }
    return activaciones


def ordenar_activaciones_por_grado(
    activaciones: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, float, dict[str, Any]]]:
    filas: list[tuple[str, float, dict[str, Any]]] = []
    for nombre, payload in activaciones.items():
        grado = float(payload.get("grado", 0.0))
        filas.append((nombre, grado, dict(payload)))
    filas.sort(key=lambda t: (-t[1], t[0]))
    return filas


def extraer_regla_dominante(
    activaciones: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    if not activaciones:
        return None, None
    ordenadas = ordenar_activaciones_por_grado(activaciones)
    nombre, _grado, payload = ordenadas[0]
    return nombre, payload


def top_reglas_activadas(
    activaciones: Mapping[str, Mapping[str, Any]],
    n: int = 5,
) -> Sequence[tuple[str, float, dict[str, Any]]]:
    ordenadas = ordenar_activaciones_por_grado(activaciones)
    return ordenadas[: max(0, int(n))]


# 3. Agregación — máximo de salidas recortadas


def agregar_salida(
    activaciones: Mapping[str, Mapping[str, Any]],
    universo_salida: np.ndarray,
    curvas_salida: Mapping[str, np.ndarray],
) -> np.ndarray:
    agregado = np.zeros_like(universo_salida, dtype=float)
    for datos in activaciones.values():
        grado = float(datos["grado"])
        if grado <= 0.0:
            continue
        etiqueta = str(datos["salida"])
        recortada = np.minimum(np.asarray(curvas_salida[etiqueta], dtype=float), grado)
        agregado = np.maximum(agregado, recortada)
    return agregado


# 4. Defuzzificación — centroide y etiqueta en salida


def defuzzificar_centroide(universo_salida: np.ndarray, agregado: np.ndarray, fallback: float = 15.0) -> float:
    universo = np.asarray(universo_salida, dtype=float)
    curva = np.asarray(agregado, dtype=float)
    area = float(np.trapezoid(curva, universo))
    if area <= 1e-9:
        return float(fallback)
    momento = float(np.trapezoid(universo * curva, universo))
    return float(momento / area)


def obtener_etiqueta_salida(valor: float, universo_salida: np.ndarray, curvas_salida: Mapping[str, np.ndarray]) -> str:
    grados = calcular_grados_salida_crisp(valor, universo_salida, curvas_salida)
    return max(grados, key=grados.get)


def calcular_grados_salida_crisp(
    valor: float,
    universo_salida: np.ndarray,
    curvas_salida: Mapping[str, np.ndarray],
) -> dict[str, float]:
    u = np.asarray(universo_salida, dtype=float)
    return {
        etiqueta: float(interpolar_membresia(u, np.asarray(curva, dtype=float), float(valor)))
        for etiqueta, curva in curvas_salida.items()
    }


def ejecutar_inferencia_mamdani(
    entradas: Mapping[str, float],
    membresias: Mapping[str, Mapping[str, Any]],
    fallback: float = 15.0,
) -> dict[str, Any]:
    grados = calcular_membresias_entrada(entradas, membresias)
    activaciones = activar_reglas(grados)
    universo_salida = np.asarray(membresias["duracion_riego"]["universo"], dtype=float)
    curvas_salida = membresias["duracion_riego"]["curvas"]
    agregado = agregar_salida(activaciones, universo_salida, curvas_salida)
    valor = defuzzificar_centroide(universo_salida, agregado, fallback=fallback)
    etiqueta = obtener_etiqueta_salida(valor, universo_salida, curvas_salida)
    grados_salida = calcular_grados_salida_crisp(valor, universo_salida, curvas_salida)
    return {
        "duracion": float(valor),
        "nivel": etiqueta,
        "grados_entrada": grados,
        "activaciones": activaciones,
        "agregado": agregado,
        "centroide": float(valor),
        "grados_salida": grados_salida,
    }
