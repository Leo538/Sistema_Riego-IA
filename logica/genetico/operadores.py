from __future__ import annotations

from typing import List, Tuple

import numpy as np


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


def reparar_cromosoma(c: np.ndarray) -> None:
    _reparar_trap_segment(c, 0, 30.0)
    c[4] = float(np.clip(c[4], 15.0, 35.0))
    c[5] = float(np.clip(c[5], 3.0, 10.0))
    _reparar_temp_alta(c)
    _reparar_trap_segment(c, 9, 60.0)
    seg = np.clip(c[13:16], 20.0, 80.0)
    seg.sort()
    c[13:16] = seg
    _reparar_hum_alta(c)


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

