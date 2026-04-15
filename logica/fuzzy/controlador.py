from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .inferencia import (
    calcular_grados_salida_crisp,
    ejecutar_inferencia_mamdani,
    extraer_regla_dominante,
    top_reglas_activadas,
)
from .membresia import (
    VARIABLES_DIFUSAS,
    construir_membresias,
    construir_membresias_desde_parametros_optimizados,
    describir_configuracion,
)


ENTRADAS_REFERENCIA = {
    "humedad_suelo": 36.0,
    "temperatura": 25.0,
    "humedad_relativa": 60.0,
    "par": 1000.0,
}

NOMBRES_ENTRADA_EXPLICACION: dict[str, str] = {
    "humedad_suelo": "humedad del suelo",
    "temperatura": "temperatura",
    "humedad_relativa": "humedad relativa",
    "par": "PAR",
}

ORDEN_ANTECEDENTES_EXPLICACION = ("humedad_suelo", "temperatura", "humedad_relativa", "par")


def _antecedentes_a_frase_natural(antecedentes: Mapping[str, str], salida: str) -> str:
    partes = [
        f"{NOMBRES_ENTRADA_EXPLICACION[var]} es {antecedentes[var]}"
        for var in ORDEN_ANTECEDENTES_EXPLICACION
        if var in antecedentes
    ]
    return f"si {' y '.join(partes)}, entonces la duración del riego es {salida}"


def obtener_regla_dominante(activaciones: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    nombre, payload = extraer_regla_dominante(activaciones)
    if nombre is None or payload is None:
        return {
            "nombre": None,
            "grado": 0.0,
            "salida": None,
            "descripcion": None,
            "antecedentes": None,
            "frase_natural": None,
        }
    antecedentes = dict(payload.get("antecedentes", {}))
    salida = str(payload.get("salida", ""))
    grado = float(payload.get("grado", 0.0))
    descripcion = str(payload.get("descripcion", "")).strip()
    return {
        "nombre": nombre,
        "grado": grado,
        "salida": salida,
        "descripcion": descripcion or None,
        "antecedentes": antecedentes,
        "frase_natural": _antecedentes_a_frase_natural(antecedentes, salida),
    }


def obtener_top_reglas(activaciones: Mapping[str, Mapping[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    filas: list[dict[str, Any]] = []
    for nombre, grado, payload in top_reglas_activadas(activaciones, n=top_n):
        antecedentes = dict(payload.get("antecedentes", {}))
        salida = str(payload.get("salida", ""))
        descripcion = str(payload.get("descripcion", "")).strip()
        filas.append(
            {
                "nombre": nombre,
                "grado": float(grado),
                "salida": salida,
                "descripcion": descripcion or None,
                "antecedentes": antecedentes,
                "frase_natural": _antecedentes_a_frase_natural(antecedentes, salida),
            }
        )
    return filas


def calcular_grados_salida(
    valor_duracion: float,
    membresias: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    universo = np.asarray(membresias["duracion_riego"]["universo"], dtype=float)
    curvas = membresias["duracion_riego"]["curvas"]
    return calcular_grados_salida_crisp(float(valor_duracion), universo, curvas)


def construir_explicacion(
    entradas: Mapping[str, float],
    regla_dominante: Mapping[str, Any],
    duracion: float,
    etiqueta_salida: str,
    grados_salida: Mapping[str, float],
) -> str:
    hs = float(entradas["humedad_suelo"])
    temp = float(entradas["temperatura"])
    hr = float(entradas["humedad_relativa"])
    par_v = float(entradas["par"])
    nombre_dom = regla_dominante.get("nombre")
    grado_dom = float(regla_dominante.get("grado") or 0.0)
    frase = regla_dominante.get("frase_natural")

    if nombre_dom and frase:
        bloque_regla = (
            f'La regla dominante fue "{frase}", con grado de activación {grado_dom:.2f} '
            f"(identificador {nombre_dom})."
        )
    else:
        bloque_regla = "No se pudo determinar una regla dominante a partir de las activaciones."

    detalle_grados = ", ".join(f"{et}={float(grados_salida[et]):.2f}" for et in sorted(grados_salida))

    return (
        f"Con humedad del suelo {hs:.1f}, temperatura {temp:.1f}, humedad relativa {hr:.1f} y PAR {par_v:.1f}, "
        f"{bloque_regla} "
        f"La salida defuzzificada por centroide fue {duracion:.1f} min. "
        f'La etiqueta dominante de duración en ese valor crisp fue "{etiqueta_salida}". '
        f"Grados en las funciones de salida (corta / media / larga): {detalle_grados}."
    )


def validar_y_recortar_entradas(
    humedad_suelo: float,
    temperatura: float,
    humedad_relativa: float,
    par: float,
) -> dict[str, float]:
    entradas = {
        "humedad_suelo": float(humedad_suelo),
        "temperatura": float(temperatura),
        "humedad_relativa": float(humedad_relativa),
        "par": float(par),
    }
    for variable, valor in list(entradas.items()):
        definicion = VARIABLES_DIFUSAS[variable]
        entradas[variable] = float(np.clip(valor, definicion.minimo, definicion.maximo))
    return entradas


@dataclass
class ControladorDifusoRiego:
    parametros_membresia: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None

    def construir_membresias(self) -> dict[str, dict[str, Any]]:
        if self.parametros_membresia is None:
            return construir_membresias()
        return construir_membresias_desde_parametros_optimizados(self.parametros_membresia)

    def evaluar(
        self,
        humedad_suelo: float,
        temperatura: float,
        humedad_relativa: float,
        par: float,
    ) -> dict[str, Any]:
        entradas = validar_y_recortar_entradas(humedad_suelo, temperatura, humedad_relativa, par)
        membresias = self.construir_membresias()
        resultado = ejecutar_inferencia_mamdani(entradas, membresias)
        activaciones = resultado["activaciones"]
        regla_dom = obtener_regla_dominante(activaciones)
        top = obtener_top_reglas(activaciones, top_n=5)
        grados_salida = resultado["grados_salida"]
        explicacion = construir_explicacion(
            entradas,
            regla_dom,
            float(resultado["duracion"]),
            str(resultado["nivel"]),
            grados_salida,
        )
        return {
            "duracion": float(resultado["duracion"]),
            "nivel": str(resultado["nivel"]),
            "activaciones": activaciones,
            "entradas": entradas,
            "configuracion": describir_configuracion(self.parametros_membresia),
            "grados_entrada": resultado["grados_entrada"],
            "grados_salida": grados_salida,
            "centroide": float(resultado["centroide"]),
            "agregado": resultado["agregado"],
            "regla_dominante": regla_dom,
            "top_reglas": top,
            "explicacion": explicacion,
        }


def calcular_duracion_riego(
    humedad_suelo: float,
    temperatura: float,
    humedad_relativa: float,
    par: float,
    parametros_membresia: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None,
) -> dict[str, Any]:
    controlador = ControladorDifusoRiego(parametros_membresia=parametros_membresia)
    return controlador.evaluar(humedad_suelo, temperatura, humedad_relativa, par)


def obtener_curvas_para_graficas(
    parametros_membresia: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None,
) -> dict[str, dict[str, Any]]:
    return ControladorDifusoRiego(parametros_membresia=parametros_membresia).construir_membresias()


def preparar_datos_comparacion(
    entradas: Mapping[str, float],
    parametros_optimizados: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None,
) -> dict[str, Any]:
    base = calcular_duracion_riego(**entradas)
    optimizado = calcular_duracion_riego(**entradas, parametros_membresia=parametros_optimizados)
    return {
        "base": base,
        "optimizado": optimizado,
        "diferencia_duracion": float(optimizado["duracion"] - base["duracion"]),
    }
