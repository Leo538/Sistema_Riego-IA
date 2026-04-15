from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from logica.fuzzy.controlador import ENTRADAS_REFERENCIA
from logica.fuzzy.reglas import consultar_reglas_por_salida, obtener_reglas_como_filas_ui
from logica.fuzzy.membresia import ORDEN_ENTRADAS, VARIABLES_DIFUSAS
from logica.fuzzy.sistema import calcular_riego, curvas_membresia_para_grafico
from logica.simulacion import ejecutar_con_genetico, generar_datos

from .cultivo import render_cultivo

COLOR_BAJA = "#1f77b4"
COLOR_MEDIA = "#2ca02c"
COLOR_ALTA = "#d62728"
COLOR_ACENTO = "#555555"

_GENES_POR_VARIABLE = 11

_LABELS_ENTRADA: Dict[str, Tuple[str, str]] = {
    "humedad_suelo": ("Humedad del suelo", "Indice suelo (0-64)"),
    "temperatura": ("Temperatura", "Temperatura (C)"),
    "humedad_relativa": ("Humedad relativa", "Humedad relativa (%)"),
    "par": ("PAR", "PAR (0-2000 umol m-2 s-1)"),
}


def _limpiar_claves_sesion_obsoletas() -> None:
    for clave in ("tipo_ui", "slider_hum", "humedad"):
        st.session_state.pop(clave, None)


def _init_session() -> None:
    ref = ENTRADAS_REFERENCIA
    if "temp" not in st.session_state:
        st.session_state.temp = float(ref["temperatura"])
    if "humedad_suelo" not in st.session_state:
        st.session_state.humedad_suelo = float(ref["humedad_suelo"])
    if "humedad_relativa" not in st.session_state:
        st.session_state.humedad_relativa = float(ref["humedad_relativa"])
    if "par" not in st.session_state:
        st.session_state.par = float(ref["par"])
    if "slider_temp" not in st.session_state:
        st.session_state.slider_temp = st.session_state.temp
    if "slider_hs" not in st.session_state:
        st.session_state.slider_hs = st.session_state.humedad_suelo
    if "slider_hr" not in st.session_state:
        st.session_state.slider_hr = st.session_state.humedad_relativa
    if "slider_par" not in st.session_state:
        st.session_state.slider_par = st.session_state.par
    if "n_gen" not in st.session_state:
        st.session_state.n_gen = 50
    if "n_pop" not in st.session_state:
        st.session_state.n_pop = 30
    if "comparacion" not in st.session_state:
        st.session_state.comparacion = None
    if "params_grafico" not in st.session_state:
        st.session_state.params_grafico = None


def _aplicar_cambios_pendientes_antes_de_sliders() -> None:
    ref = ENTRADAS_REFERENCIA
    if st.session_state.pop("_pendiente_reinicio", False):
        st.session_state.slider_temp = float(ref["temperatura"])
        st.session_state.slider_hs = float(ref["humedad_suelo"])
        st.session_state.slider_hr = float(ref["humedad_relativa"])
        st.session_state.slider_par = float(ref["par"])
        st.session_state.temp = float(ref["temperatura"])
        st.session_state.humedad_suelo = float(ref["humedad_suelo"])
        st.session_state.humedad_relativa = float(ref["humedad_relativa"])
        st.session_state.par = float(ref["par"])
        st.session_state.n_gen = 50
        st.session_state.n_pop = 30
        st.session_state.comparacion = None
        st.session_state.params_grafico = None
        if "ultimo_aviso" in st.session_state:
            del st.session_state.ultimo_aviso
        return
    datos_rand = st.session_state.pop("_pendiente_datos_aleatorios", None)
    if datos_rand is not None:
        st.session_state.slider_temp = float(datos_rand["temperatura"])
        st.session_state.slider_hs = float(datos_rand["humedad_suelo"])
        st.session_state.slider_hr = float(datos_rand["humedad_relativa"])
        st.session_state.slider_par = float(datos_rand["par"])
        st.session_state.temp = float(datos_rand["temperatura"])
        st.session_state.humedad_suelo = float(datos_rand["humedad_suelo"])
        st.session_state.humedad_relativa = float(datos_rand["humedad_relativa"])
        st.session_state.par = float(datos_rand["par"])


def _fig_membresia(
    titulo: str,
    x: np.ndarray,
    curvas: Dict[str, np.ndarray],
    x_actual: Optional[float],
    xlabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.2), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(x, curvas["baja"], color=COLOR_BAJA, label="Baja", linewidth=2)
    ax.plot(x, curvas["media"], color=COLOR_MEDIA, label="Media", linewidth=2)
    ax.plot(x, curvas["alta"], color=COLOR_ALTA, label="Alta", linewidth=2)
    if x_actual is not None:
        ax.axvline(x_actual, color=COLOR_ACENTO, linestyle="--", linewidth=1.5, label="Valor actual")
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Grado de pertenencia")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


@lru_cache(maxsize=1)
def _filas_base_reglas_cache() -> tuple[dict[str, Any], ...]:
    return tuple(obtener_reglas_como_filas_ui())


def _fig_agregacion_centroide(
    x: np.ndarray,
    curvas_salida_ui: Dict[str, np.ndarray],
    agregado: np.ndarray,
    centroide: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 3.8), facecolor="white")
    ax.set_facecolor("#fafafa")
    x = np.asarray(x, dtype=float)
    agg = np.asarray(agregado, dtype=float)
    ax.plot(x, curvas_salida_ui["bajo"], color=COLOR_BAJA, linestyle=":", linewidth=1.4, alpha=0.85, label="Corta (forma base)")
    ax.plot(x, curvas_salida_ui["medio"], color=COLOR_MEDIA, linestyle=":", linewidth=1.4, alpha=0.85, label="Media (forma base)")
    ax.plot(x, curvas_salida_ui["alto"], color=COLOR_ALTA, linestyle=":", linewidth=1.4, alpha=0.85, label="Larga (forma base)")
    ax.plot(x, agg, color="#5e35b1", linewidth=2.4, label="Agregado (máximo)")
    ax.axvline(float(centroide), color="#212121", linestyle="-", linewidth=2.0, label=f"Centroide = {float(centroide):.2f} min")
    ax.set_title("Salida: agregación y defuzzificación (centroide)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Duración (min)")
    ax.set_ylabel("Grado")
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_riego_salida(curvas: Dict[str, np.ndarray], x: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.2), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(x, curvas["bajo"], color=COLOR_BAJA, label="Corta (bajo)", linewidth=2)
    ax.plot(x, curvas["medio"], color=COLOR_MEDIA, label="Media (medio)", linewidth=2)
    ax.plot(x, curvas["alto"], color=COLOR_ALTA, label="Larga (alto)", linewidth=2)
    ax.set_title("Salida: duracion de riego", fontsize=12, fontweight="bold")
    ax.set_xlabel("Duracion (min), universo 0-30")
    ax.set_ylabel("Grado de pertenencia")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _nota_tabla_genes_entrada(optimizado: bool) -> str:
    if optimizado:
        return "AG ajusto estos parametros (11 genes por variable de entrada)."
    return "Ejecute el AG para ver la comparacion"


def _fmt_tpl(tpl: tuple[float, ...]) -> str:
    return str([round(float(x), 2) for x in tpl])


def _markdown_tabla_variable(
    params_plot: Optional[np.ndarray],
    indice_variable: int,
) -> str:
    clave = ORDEN_ENTRADAS[indice_variable]
    definicion = VARIABLES_DIFUSAS[clave]
    et = definicion.etiquetas
    orig_b = _fmt_tpl(tuple(et["baja"]))
    orig_m = _fmt_tpl(tuple(et["media"]))
    orig_a = _fmt_tpl(tuple(et["alta"]))
    if params_plot is None:
        ag_b = ag_m = ag_a = "— sin optimizar"
    else:
        p = np.asarray(params_plot, dtype=float).flatten()
        s = indice_variable * _GENES_POR_VARIABLE
        bloque = p[s : s + _GENES_POR_VARIABLE]
        ag_b = str([round(float(x), 2) for x in bloque[0:4]])
        ag_m = str([round(float(x), 2) for x in bloque[4:7]])
        ag_a = str([round(float(x), 2) for x in bloque[7:11]])
    return f"""
| Categoria | Forma | Parametros originales | Parametros AG |
| :--- | :--- | :--- | :--- |
| Baja | Trapezoidal | {orig_b} | {ag_b} |
| Media | Triangular | {orig_m} | {ag_m} |
| Alta | Trapezoidal | {orig_a} | {ag_a} |

{_nota_tabla_genes_entrada(params_plot is not None)}
"""


def _markdown_tabla_riego() -> str:
    et = VARIABLES_DIFUSAS["duracion_riego"].etiquetas
    return f"""
| Categoria | Forma | Parametros originales | Parametros AG |
| :--- | :--- | :--- | :--- |
| Corta | Trapezoidal | {_fmt_tpl(tuple(et["corta"]))} | No optimizado |
| Media | Triangular | {_fmt_tpl(tuple(et["media"]))} | No optimizado |
| Larga | Trapezoidal | {_fmt_tpl(tuple(et["larga"]))} | No optimizado |

La salida no es optimizada por el AG
"""


def _nivel_color_html(nivel: str) -> str:
    n = nivel.lower()
    if n == "bajo":
        return '<span style="color:#2ca02c;font-weight:800;font-size:1.4rem;">BAJO</span>'
    if n == "medio":
        return '<span style="color:#ff9800;font-weight:800;font-size:1.4rem;">MEDIO</span>'
    return '<span style="color:#d62728;font-weight:800;font-size:1.4rem;">ALTO</span>'


def ejecutar_interfaz() -> None:
    st.set_page_config(page_title="Riego inteligente", layout="wide", initial_sidebar_state="collapsed")
    _limpiar_claves_sesion_obsoletas()
    _init_session()
    _aplicar_cambios_pendientes_antes_de_sliders()

    st.title("Sistema inteligente de riego automatizado")
    st.caption(
        "Lógica difusa Mamdani (4 entradas, 81 reglas): membresías con trapecios en baja/alta y triángulos en media; "
        "optimización genética solo sobre parámetros de entrada."
    )

    tab_panel, tab_cultivo = st.tabs(["Panel de Control", "Cultivo"])

    with tab_panel:
        col_izq, col_der = st.columns([1, 1.35])

        with col_izq:
            st.subheader("1. Entradas")
            hs = st.slider(
                _LABELS_ENTRADA["humedad_suelo"][0] + " (0-64)",
                0.0,
                64.0,
                key="slider_hs",
                step=0.5,
            )
            temp = st.slider(
                _LABELS_ENTRADA["temperatura"][0] + " (0-50 C)",
                0.0,
                50.0,
                key="slider_temp",
                step=0.5,
            )
            hr = st.slider(
                _LABELS_ENTRADA["humedad_relativa"][0] + " (0-100 %)",
                0.0,
                100.0,
                key="slider_hr",
                step=0.5,
            )
            par_v = st.slider(
                _LABELS_ENTRADA["par"][0] + " (0-2000)",
                0.0,
                2000.0,
                key="slider_par",
                step=10.0,
            )
            st.session_state.temp = temp
            st.session_state.humedad_suelo = hs
            st.session_state.humedad_relativa = hr
            st.session_state.par = par_v

            if st.button("Generar datos aleatorios", key="btn_rand"):
                st.session_state._pendiente_datos_aleatorios = generar_datos()
                st.rerun()

            st.subheader("2. Configuración del modelo")
            st.caption(
                "Las funciones de pertenencia siguen el diseño del backend: **trapecios** para las etiquetas "
                "**baja** y **alta**, y **triángulos** para la etiqueta **media** (cuatro variables de entrada y la salida). "
                "El algoritmo genético ajusta 11 parámetros por variable de entrada (4+3+4), sin cambiar ese esquema."
            )

            n_gen = st.slider("Numero de generaciones", 30, 100, int(st.session_state.n_gen), 1)
            n_pop = st.slider("Tamano de poblacion", 20, 50, int(st.session_state.n_pop), 1)
            st.session_state.n_gen = n_gen
            st.session_state.n_pop = n_pop

            st.subheader("3. Acciones")
            if st.button("Ver resultado actual", type="primary"):
                st.session_state.ultimo_aviso = (
                    "Resultado actual mostrado (se actualiza tambien al mover los sliders)."
                )
            if st.button("Optimizar con Algoritmo Genetico"):
                with st.spinner("Evolucionando poblacion y ajustando membresias..."):
                    try:
                        comp = ejecutar_con_genetico(
                            hs,
                            temp,
                            hr,
                            par_v,
                            n_generaciones=n_gen,
                            tam_poblacion=n_pop,
                        )
                        st.session_state.comparacion = comp
                        st.session_state.params_grafico = np.array(comp["mejor_individuo"], dtype=float)
                    except Exception as e:
                        st.error(str(e))
                        st.session_state.comparacion = None
                        st.session_state.params_grafico = None
                if st.session_state.comparacion is not None:
                    st.success("Optimizacion completada.")
            if st.button("Reiniciar"):
                st.session_state._pendiente_reinicio = True
                st.rerun()

            if getattr(st.session_state, "ultimo_aviso", None):
                st.info(st.session_state.ultimo_aviso)

        params_plot = st.session_state.params_grafico
        try:
            datos = curvas_membresia_para_grafico(parametros=params_plot)
            live = calcular_riego(
                temp,
                hs,
                parametros=None,
                humedad_relativa=hr,
                par=par_v,
            )
        except Exception as e:
            st.error(f"No se pudo evaluar el sistema difuso: {e}")
            return

        with col_der:
            st.subheader("4. Funciones de membresía")
            if params_plot is not None:
                st.caption(
                    "Curvas con parámetros optimizados por el AG (mismo esquema trapecio/triángulo). "
                    "Tras Reiniciar vuelven al diseño base."
                )
            else:
                st.caption(
                    "Curvas con parámetros por defecto: baja/alta = trapecios, media = triángulo. "
                    "Optimice para ver el ajuste evolutivo."
                )

            graficas_entrada = [
                (
                    _LABELS_ENTRADA["humedad_suelo"][0],
                    datos["humedad_x"],
                    datos["humedad"],
                    hs,
                    _LABELS_ENTRADA["humedad_suelo"][1],
                    0,
                ),
                (
                    _LABELS_ENTRADA["temperatura"][0],
                    datos["temp_x"],
                    datos["temp"],
                    temp,
                    _LABELS_ENTRADA["temperatura"][1],
                    1,
                ),
                (
                    _LABELS_ENTRADA["humedad_relativa"][0],
                    datos["humedad_rel_x"],
                    datos["humedad_rel"],
                    hr,
                    _LABELS_ENTRADA["humedad_relativa"][1],
                    2,
                ),
                (
                    _LABELS_ENTRADA["par"][0],
                    datos["par_x"],
                    datos["par"],
                    par_v,
                    _LABELS_ENTRADA["par"][1],
                    3,
                ),
            ]

            for titulo, x_arr, curvas, x0, xlab, idx_var in graficas_entrada:
                f = _fig_membresia(titulo, x_arr, curvas, x0, xlab)
                st.pyplot(f, clear_figure=True)
                plt.close(f)
                st.markdown(_markdown_tabla_variable(params_plot, idx_var))

            f3 = _fig_riego_salida(datos["riego"], datos["riego_x"])
            st.pyplot(f3, clear_figure=True)
            plt.close(f3)
            st.markdown(_markdown_tabla_riego())

            st.subheader("Inferencia Mamdani: reglas y agregación")
            st.caption(
                "Trazas generadas en el backend (mismas activaciones que el motor difuso). "
                "La curva violeta es el máximo de los recortes; la línea vertical marca el centroide."
            )

            dom = live.get("regla_dominante") or {}
            if dom.get("nombre"):
                st.markdown("**Regla dominante**")
                st.write(f"**Identificador:** `{dom['nombre']}`")
                st.write(f"**Grado de activación:** {float(dom['grado']):.4f}")
                st.write(f"**Salida lingüística de la regla:** {dom.get('salida')}")
                if dom.get("frase_natural"):
                    st.caption(dom["frase_natural"])
                if dom.get("descripcion"):
                    st.caption(dom["descripcion"])
            else:
                st.info("No hay información de regla dominante (activaciones vacías o no disponibles).")

            tops = live.get("top_reglas") or []
            if tops:
                st.markdown("**Top reglas activadas**")
                tabla_top = [
                    {
                        "nombre": r["nombre"],
                        "grado": round(float(r["grado"]), 4),
                        "salida": r["salida"],
                        "frase": r.get("frase_natural") or "",
                    }
                    for r in tops
                ]
                st.dataframe(tabla_top, hide_index=True, use_container_width=True)

            if live.get("explicacion"):
                st.markdown("**Explicación automática**")
                st.info(str(live["explicacion"]))

            agg = live.get("agregado")
            cen = float(live.get("centroide", live.get("duracion", 0.0)))
            if agg is not None:
                f_agg = _fig_agregacion_centroide(
                    datos["riego_x"],
                    datos["riego"],
                    np.asarray(agg, dtype=float),
                    cen,
                )
                st.pyplot(f_agg, clear_figure=True)
                plt.close(f_agg)

            with st.expander("Base completa de reglas (81)", expanded=False):
                st.caption("Misma base que en `logica/fuzzy/reglas.py`; tabla generada en el backend.")
                filtro_salida = st.selectbox(
                    "Filtrar por etiqueta de salida",
                    ("Todas", "corta", "media", "larga"),
                    index=0,
                    key="filtro_tabla_reglas",
                )
                if filtro_salida == "Todas":
                    filas_reglas = list(_filas_base_reglas_cache())
                else:
                    filas_reglas = obtener_reglas_como_filas_ui(consultar_reglas_por_salida(filtro_salida))
                st.dataframe(filas_reglas, hide_index=True, height=420, use_container_width=True)

            st.subheader("5. Resultado en tiempo real (sin optimizar genes)")
            dur = float(live["duracion"])
            st.metric("Duracion de riego (defuzz.)", f"{dur:.2f} min")
            st.markdown(_nivel_color_html(live["nivel"]), unsafe_allow_html=True)
            st.progress(min(max(dur / 30.0, 0.0), 1.0))

            st.subheader("Reporte PDF")
            try:
                from logica.reporte_pdf import generar_reporte_sesion_pdf

                pdf_bytes = generar_reporte_sesion_pdf(
                    humedad_suelo=float(hs),
                    temperatura=float(temp),
                    humedad_relativa=float(hr),
                    par=float(par_v),
                    n_generaciones=int(st.session_state.n_gen),
                    n_poblacion=int(st.session_state.n_pop),
                    resultado_sin=live,
                    parametros_grafico=st.session_state.params_grafico,
                    comparacion=st.session_state.comparacion,
                )
                nombre_pdf = f"reporte_riego_{datetime.now():%Y%m%d_%H%M%S}.pdf"
                st.download_button(
                    "Descargar reporte PDF",
                    data=pdf_bytes,
                    file_name=nombre_pdf,
                    mime="application/pdf",
                    help="Entradas, resultados, reglas activadas, membresias y evolucion del fitness si hubo AG.",
                )
            except ModuleNotFoundError:
                st.warning(
                    "Para el PDF falta reportlab (y Pillow). Ejecute: python -m pip install reportlab Pillow"
                )
            except Exception as ex:
                st.warning(f"No se pudo generar el PDF: {ex}")

            comp = st.session_state.comparacion
            if comp is not None:
                st.subheader("6. Comparacion antes / despues del AG")
                sin_o = comp["resultado_sin_optimizar"]
                con_o = comp["resultado_final"]
                fit = comp["mejor_fitness"]
                fit_norm = 1.0 / (1.0 + float(fit)) if np.isfinite(float(fit)) and float(fit) >= 0.0 else 0.0
                aceptada = bool(comp.get("se_acepta_optimizacion", False))
                motivo = str(comp.get("motivo_decision", "")).strip() or "Sin motivo disponible."
                tabla = f"""
| Metrica | Sin optimizar | Con genetico |
| :--- | :---: | :---: |
| Duracion (min) | {sin_o['agua']:.2f} | {con_o['agua']:.2f} |
| Nivel | {sin_o['nivel']} | {con_o['nivel']} |
| Fitness final | — | {fit:.4f} |
| Fitness normalizado (visual) | — | {fit_norm:.6f} |
"""
                st.markdown(tabla)
                st.markdown(f"**Optimizacion aceptada:** {'Si' if aceptada else 'No'}")
                st.caption(f"Motivo: {motivo}")

                st.subheader("7. Evolucion del fitness")
                hist = comp["historial"]
                fig_e, ax_e = plt.subplots(figsize=(6.5, 3.4), facecolor="white")
                ax_e.set_facecolor("#fafafa")
                ax_e.plot(range(1, len(hist) + 1), hist, color=COLOR_BAJA, linewidth=2, marker="o", markersize=3)
                ax_e.set_xlabel("Generacion")
                ax_e.set_ylabel("Mejor fitness acumulado")
                ax_e.set_title("Evolucion del mejor fitness (menor es mejor)", fontsize=12, fontweight="bold")
                ax_e.grid(True, alpha=0.3)
                fig_e.tight_layout()
                st.pyplot(fig_e, clear_figure=True)
                plt.close(fig_e)

            with st.expander("Detalle técnico (activaciones y trazas)"):
                traza = {
                    "duracion": live.get("duracion"),
                    "nivel_difuso": live.get("nivel_difuso"),
                    "centroide": live.get("centroide"),
                    "entradas": live.get("entradas"),
                    "grados_entrada": live.get("grados_entrada"),
                    "grados_salida": live.get("grados_salida"),
                    "regla_dominante": live.get("regla_dominante"),
                    "top_reglas": live.get("top_reglas"),
                    "explicacion": live.get("explicacion"),
                    "activaciones": live.get("activaciones"),
                }
                st.json(traza)

    with tab_cultivo:
        render_cultivo()
