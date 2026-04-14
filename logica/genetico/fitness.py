from __future__ import annotations

import numpy as np

from ..fuzzy.sistema import calcular_riego


ALPHA = 0.7
BETA = 0.3
HUMEDAD_IDEAL = 60.0


def _penalizar_incoherencia(cromosoma: np.ndarray) -> float:
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

