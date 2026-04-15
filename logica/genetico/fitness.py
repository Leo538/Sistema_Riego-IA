from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np

from ..fuzzy.controlador import calcular_duracion_riego
from ..fuzzy.membresia import construir_membresias, interpolar_membresia
from ..fuzzy.reglas import ORDEN_VARIABLES_REGLA, PUNTAJES_REGLAS

from .cromosoma import decodificar_cromosoma, reparar_cromosoma


ESCENARIOS_REFERENCIA = [
    {
        "nombre": "estres_hidrico_alto",
        "entradas": {"humedad_suelo": 10.0, "temperatura": 38.0, "humedad_relativa": 25.0, "par": 1800.0},
        "duracion_esperada": 27.0,
        "nivel_esperado": "larga",
    },
    {
        "nombre": "condicion_intermedia",
        "entradas": {"humedad_suelo": 35.0, "temperatura": 25.0, "humedad_relativa": 60.0, "par": 1000.0},
        "duracion_esperada": 15.0,
        "nivel_esperado": "media",
    },
    {
        "nombre": "baja_demanda",
        "entradas": {"humedad_suelo": 58.0, "temperatura": 14.0, "humedad_relativa": 90.0, "par": 250.0},
        "duracion_esperada": 3.0,
        "nivel_esperado": "corta",
    },
    {
        "nombre": "suelo_seco_temperatura_media",
        "entradas": {"humedad_suelo": 18.0, "temperatura": 24.0, "humedad_relativa": 55.0, "par": 900.0},
        "duracion_esperada": 20.0,
        "nivel_esperado": "media",
    },
    {
        "nombre": "demanda_extrema_alta",
        "entradas": {"humedad_suelo": 0.0, "temperatura": 50.0, "humedad_relativa": 0.0, "par": 2000.0},
        "duracion_esperada": 26.0,
        "nivel_esperado": "larga",
    },
]

PARES_ESTABILIDAD = [
    (
        {"humedad_suelo": 30.0, "temperatura": 24.0, "humedad_relativa": 58.0, "par": 980.0},
        {"humedad_suelo": 31.0, "temperatura": 24.5, "humedad_relativa": 57.0, "par": 1010.0},
    ),
    (
        {"humedad_suelo": 48.0, "temperatura": 28.0, "humedad_relativa": 62.0, "par": 1200.0},
        {"humedad_suelo": 47.0, "temperatura": 27.5, "humedad_relativa": 63.0, "par": 1180.0},
    ),
]

PARES_TENDENCIA = [
    (
        {"humedad_suelo": 55.0, "temperatura": 20.0, "humedad_relativa": 80.0, "par": 600.0},
        {"humedad_suelo": 15.0, "temperatura": 35.0, "humedad_relativa": 30.0, "par": 1600.0},
    ),
    (
        {"humedad_suelo": 45.0, "temperatura": 22.0, "humedad_relativa": 70.0, "par": 700.0},
        {"humedad_suelo": 28.0, "temperatura": 30.0, "humedad_relativa": 45.0, "par": 1300.0},
    ),
]


# 5. Función de fitness (escenarios, penalizaciones, caso local)


def penalizar_configuracion(cromosoma: Iterable[float]) -> float:
    genes = reparar_cromosoma(cromosoma)
    penalizacion = 0.0
    for inicio in range(0, len(genes), 11):
        baja = genes[inicio : inicio + 4]
        media = genes[inicio + 4 : inicio + 7]
        alta = genes[inicio + 7 : inicio + 11]
        penalizacion += max(0.0, baja[2] - media[1]) ** 2
        penalizacion += max(0.0, media[1] - alta[1]) ** 2
        penalizacion += max(0.0, media[0] - media[1]) ** 2
        penalizacion += max(0.0, media[1] - media[2]) ** 2
        ancho_media = media[2] - media[0]
        if ancho_media < 5.0:
            penalizacion += (5.0 - ancho_media) * 10.0
    return float(penalizacion)


def evaluar_escenarios(cromosoma: Iterable[float]) -> tuple[float, list[dict[str, float | str]]]:
    parametros = decodificar_cromosoma(cromosoma)
    errores = 0.0
    detalles: list[dict[str, float | str]] = []
    for escenario in ESCENARIOS_REFERENCIA:
        resultado = calcular_duracion_riego(**escenario["entradas"], parametros_membresia=parametros)
        duracion = float(resultado["duracion"])
        error = (duracion - float(escenario["duracion_esperada"])) ** 2
        if resultado["nivel"] != escenario["nivel_esperado"]:
            error += 25.0
        errores += error
        detalles.append(
            {
                "nombre": escenario["nombre"],
                "duracion_obtenida": duracion,
                "duracion_esperada": float(escenario["duracion_esperada"]),
                "error": float(error),
            }
        )
    return float(errores), detalles


def penalizar_estabilidad(cromosoma: Iterable[float]) -> float:
    parametros = decodificar_cromosoma(cromosoma)
    penalizacion = 0.0
    for caso_a, caso_b in PARES_ESTABILIDAD:
        a = calcular_duracion_riego(**caso_a, parametros_membresia=parametros)["duracion"]
        b = calcular_duracion_riego(**caso_b, parametros_membresia=parametros)["duracion"]
        penalizacion += max(0.0, abs(float(a) - float(b)) - 2.0) ** 2
    return float(penalizacion)


def penalizar_tendencia(cromosoma: Iterable[float]) -> float:
    parametros = decodificar_cromosoma(cromosoma)
    penalizacion = 0.0
    for caso_menor, caso_mayor in PARES_TENDENCIA:
        menor = float(calcular_duracion_riego(**caso_menor, parametros_membresia=parametros)["duracion"])
        mayor = float(calcular_duracion_riego(**caso_mayor, parametros_membresia=parametros)["duracion"])
        if mayor <= menor:
            penalizacion += (menor - mayor + 1.0) ** 2 * 20.0
    return float(penalizacion)


def inferir_etiqueta_objetivo_local(entradas_actuales: Mapping[str, float]) -> str:
    membresias_base = construir_membresias()
    etiquetas_dominantes: dict[str, str] = {}

    for var in ORDEN_VARIABLES_REGLA:
        valor = float(entradas_actuales[var])
        universo = np.asarray(membresias_base[var]["universo"], dtype=float)
        curvas = membresias_base[var]["curvas"]
        grados = {
            etiqueta: float(interpolar_membresia(universo, np.asarray(curva, dtype=float), valor))
            for etiqueta, curva in curvas.items()
        }
        etiquetas_dominantes[var] = max(grados, key=grados.get)

    puntaje = int(sum(PUNTAJES_REGLAS[var][etiquetas_dominantes[var]] for var in ORDEN_VARIABLES_REGLA))
    if puntaje <= 2:
        return "corta"
    if puntaje <= 5:
        return "media"
    return "larga"


def evaluar_caso_actual(cromosoma: Iterable[float], entradas_actuales: Mapping[str, float]) -> float:
    parametros = decodificar_cromosoma(cromosoma)
    resultado = calcular_duracion_riego(**entradas_actuales, parametros_membresia=parametros)
    dur = float(resultado["duracion"])

    etiqueta_objetivo = inferir_etiqueta_objetivo_local(entradas_actuales)

    centros = {"corta": 5.0, "media": 15.0, "larga": 26.0}
    centro = float(centros.get(etiqueta_objetivo, 15.0))
    error = (dur - centro) ** 2

    etiqueta_defuzz = str(resultado.get("nivel") or "")
    if etiqueta_defuzz and etiqueta_defuzz != etiqueta_objetivo:
        error += 10.0
    return float(error)


def evaluar_fitness(
    cromosoma: Iterable[float],
    *,
    entradas_actuales: Optional[Mapping[str, float]] = None,
    peso_local: float = 2.0,
) -> float:
    try:
        error_escenarios, _ = evaluar_escenarios(cromosoma)
        penalizacion = penalizar_configuracion(cromosoma)
        estabilidad = penalizar_estabilidad(cromosoma)
        tendencia = penalizar_tendencia(cromosoma)
        local = 0.0
        if entradas_actuales is not None:
            local = evaluar_caso_actual(cromosoma, entradas_actuales)
        w = float(max(0.0, min(10.0, peso_local)))
        return float(error_escenarios + w * local + penalizacion + estabilidad * 3.0 + tendencia)
    except Exception:
        return float("inf")
