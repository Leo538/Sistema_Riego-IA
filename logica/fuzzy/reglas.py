from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, List, Tuple


PUNTAJES_REGLAS = {
    "humedad_suelo": {"baja": 2, "media": 1, "alta": 0},
    "temperatura": {"baja": 0, "media": 1, "alta": 2},
    "humedad_relativa": {"alta": 0, "media": 1, "baja": 2},
    "par": {"baja": 0, "media": 1, "alta": 2},
}

ORDEN_VARIABLES_REGLA = ("humedad_suelo", "temperatura", "humedad_relativa", "par")
ORDEN_ETIQUETAS = ("baja", "media", "alta")
VARIABLE_SALIDA = "duracion_riego"


# 2. Inferencia — base de reglas Mamdani (81 reglas)


@dataclass
class FuzzyRule:
    name: str
    antecedents: List[Tuple[str, str]]
    consequent: Tuple[str, str]
    weight: float = 1.0
    description: str = ""
    score: int = 0

    def __repr__(self) -> str:
        ant_str = " Y ".join(f"{var} ES {label}" for var, label in self.antecedents)
        con_str = f"{self.consequent[0]} ES {self.consequent[1]}"
        return f"{self.name}: SI {ant_str} ENTONCES {con_str} (w={self.weight:.2f})"


class RuleSet:
    def __init__(self) -> None:
        self.rules: List[FuzzyRule] = []

    def add_rule(
        self,
        antecedents: List[Tuple[str, str]],
        consequent: Tuple[str, str],
        weight: float = 1.0,
        description: str = "",
        name: str | None = None,
        score: int = 0,
    ) -> RuleSet:
        rule_name = name if name is not None else f"regla_{len(self.rules) + 1:02d}"
        self.rules.append(
            FuzzyRule(
                name=rule_name,
                antecedents=antecedents,
                consequent=consequent,
                weight=weight,
                description=description,
                score=score,
            )
        )
        return self

    def get_rules(self) -> List[FuzzyRule]:
        return self.rules

    @property
    def num_rules(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"RuleSet({self.num_rules} reglas)"


def _etiqueta_salida_desde_puntaje(puntaje: int) -> str:
    if puntaje <= 2:
        return "corta"
    if puntaje <= 5:
        return "media"
    return "larga"


def create_default_rule_base(output_name: str = VARIABLE_SALIDA) -> RuleSet:
    rb = RuleSet()
    for indice, combinacion in enumerate(product(ORDEN_ETIQUETAS, repeat=4), start=1):
        antecedents = list(zip(ORDEN_VARIABLES_REGLA, combinacion))
        puntaje = sum(PUNTAJES_REGLAS[variable][etiqueta] for variable, etiqueta in antecedents)
        salida = _etiqueta_salida_desde_puntaje(puntaje)
        por_variable = dict(antecedents)
        descripcion = (
            f"Si humedad_suelo es {por_variable['humedad_suelo']} y temperatura es {por_variable['temperatura']} "
            f"y humedad_relativa es {por_variable['humedad_relativa']} y par es {por_variable['par']}, "
            f"entonces {output_name} es {salida}."
        )
        rb.add_rule(
            antecedents=antecedents,
            consequent=(output_name, salida),
            weight=1.0,
            description=descripcion,
            name=f"regla_{indice:02d}",
            score=puntaje,
        )
    return rb


ReglaDifusa = FuzzyRule


class BaseReglasDifusas:
    def __init__(self, rule_set: RuleSet | None = None) -> None:
        self._rule_set = rule_set or create_default_rule_base()

    @property
    def reglas(self) -> List[FuzzyRule]:
        return self._rule_set.rules

    def agregar_regla(
        self,
        nombre: str,
        antecedentes: list[tuple[str, str]],
        consecuente: tuple[str, str],
        puntaje: int,
        peso: float = 1.0,
        descripcion: str = "",
    ) -> BaseReglasDifusas:
        self._rule_set.add_rule(
            antecedents=antecedentes,
            consequent=consecuente,
            weight=peso,
            description=descripcion,
            name=nombre,
            score=puntaje,
        )
        return self

    def obtener_reglas(self) -> list[FuzzyRule]:
        return self._rule_set.get_rules()

    @property
    def numero_reglas(self) -> int:
        return self._rule_set.num_rules

    def __repr__(self) -> str:
        return f"BaseReglasDifusas({self.numero_reglas} reglas)"


def generar_base_reglas() -> BaseReglasDifusas:
    return BaseReglasDifusas(create_default_rule_base())


BASE_REGLAS = generar_base_reglas()


def obtener_reglas() -> list[FuzzyRule]:
    return BASE_REGLAS.obtener_reglas()


def consultar_reglas_por_salida(etiqueta_salida: str) -> list[FuzzyRule]:
    return [regla for regla in BASE_REGLAS.obtener_reglas() if regla.consequent[1] == etiqueta_salida]


def regla_a_texto_claro(regla: FuzzyRule) -> str:
    if regla.description.strip():
        return regla.description.strip()
    por_var = dict(regla.antecedents)
    partes = [
        f"humedad_suelo es {por_var['humedad_suelo']}",
        f"temperatura es {por_var['temperatura']}",
        f"humedad_relativa es {por_var['humedad_relativa']}",
        f"par es {por_var['par']}",
    ]
    salida = regla.consequent[1]
    return (
        f"Si {' y '.join(partes)}, entonces {VARIABLE_SALIDA} es {salida}."
    )


def buscar_regla_por_nombre(nombre: str) -> FuzzyRule | None:
    for regla in BASE_REGLAS.obtener_reglas():
        if regla.name == nombre:
            return regla
    return None


def regla_a_fila_tabla(regla: FuzzyRule) -> dict[str, Any]:
    por_variable = dict(regla.antecedents)
    return {
        "nombre": regla.name,
        "humedad_suelo": por_variable["humedad_suelo"],
        "temperatura": por_variable["temperatura"],
        "humedad_relativa": por_variable["humedad_relativa"],
        "par": por_variable["par"],
        "salida": regla.consequent[1],
        "puntaje": int(regla.score),
        "descripcion": regla_a_texto_claro(regla),
    }


def obtener_reglas_como_filas_ui(reglas: list[FuzzyRule] | None = None) -> list[dict[str, Any]]:
    lista = reglas if reglas is not None else obtener_reglas()
    return [regla_a_fila_tabla(r) for r in lista]


def listar_reglas_como_texto(reglas: list[FuzzyRule] | None = None) -> list[str]:
    lista = reglas if reglas is not None else obtener_reglas()
    return [regla_a_texto_claro(r) for r in lista]
