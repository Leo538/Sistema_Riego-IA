"""
Interfaz Streamlit del sistema de riego inteligente.
Solo presentación: toda la lógica reside en el paquete `logica`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from logica.fuzzy_system import calcular_riego, curvas_membresia_para_grafico
from logica.simulacion import ejecutar_con_genetico, generar_datos

from .cultivo_mora import render_cultivo_mora

# Paleta coherente con la especificación
COLOR_BAJA = "#1f77b4"
COLOR_MEDIA = "#2ca02c"
COLOR_ALTA = "#d62728"
COLOR_ACENTO = "#555555"


def _init_session() -> None:
    if "temp" not in st.session_state:
        st.session_state.temp = 25.0
    if "humedad" not in st.session_state:
        st.session_state.humedad = 50.0
    if "slider_temp" not in st.session_state:
        st.session_state.slider_temp = st.session_state.temp
    if "slider_hum" not in st.session_state:
        st.session_state.slider_hum = st.session_state.humedad
    if "tipo_ui" not in st.session_state:
        st.session_state.tipo_ui = "Triangular"
    if "n_gen" not in st.session_state:
        st.session_state.n_gen = 50
    if "n_pop" not in st.session_state:
        st.session_state.n_pop = 30
    if "comparacion" not in st.session_state:
        st.session_state.comparacion = None
    if "params_grafico" not in st.session_state:
        st.session_state.params_grafico = None


def _aplicar_cambios_pendientes_antes_de_sliders() -> None:
    """
    Streamlit no permite asignar st.session_state[clave] de un slider
    después de instanciar el widget. Si hubo Reiniciar o datos aleatorios,
    se guardó una señal: aquí (antes de crear los sliders) aplicamos valores.
    """
    if st.session_state.pop("_pendiente_reinicio", False):
        st.session_state.slider_temp = 25.0
        st.session_state.slider_hum = 50.0
        st.session_state.temp = 25.0
        st.session_state.humedad = 50.0
        st.session_state.tipo_ui = "Triangular"
        st.session_state.n_gen = 50
        st.session_state.n_pop = 30
        st.session_state.comparacion = None
        st.session_state.params_grafico = None
        if "ultimo_aviso" in st.session_state:
            del st.session_state.ultimo_aviso
        return
    datos_rand = st.session_state.pop("_pendiente_datos_aleatorios", None)
    if datos_rand is not None:
        t = float(datos_rand["temperatura"])
        h = float(datos_rand["humedad"])
        st.session_state.slider_temp = t
        st.session_state.slider_hum = h
        st.session_state.temp = t
        st.session_state.humedad = h


def _map_tipo(ui_val: str) -> str:
    return "gaussiana" if ui_val == "Gaussiana" else "triangular"


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
    ax.set_ylabel("Grado de pertenencia μ")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_riego_salida(curvas: Dict[str, np.ndarray], x: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.2), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(x, curvas["bajo"], color=COLOR_BAJA, label="Bajo", linewidth=2)
    ax.plot(x, curvas["medio"], color=COLOR_MEDIA, label="Medio", linewidth=2)
    ax.plot(x, curvas["alto"], color=COLOR_ALTA, label="Alto", linewidth=2)
    ax.set_title("Funciones de membresía — Riego (salida)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Litros de agua")
    ax.set_ylabel("Grado de pertenencia μ")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _nivel_color_html(nivel: str) -> str:
    n = nivel.lower()
    if n == "bajo":
        return '<span style="color:#2ca02c;font-weight:800;font-size:1.4rem;">BAJO</span>'
    if n == "medio":
        return '<span style="color:#ff9800;font-weight:800;font-size:1.4rem;">MEDIO</span>'
    return '<span style="color:#d62728;font-weight:800;font-size:1.4rem;">ALTO</span>'


def ejecutar_interfaz() -> None:
    st.set_page_config(page_title="Riego inteligente", layout="wide", initial_sidebar_state="collapsed")
    _init_session()
    _aplicar_cambios_pendientes_antes_de_sliders()

    st.title("Sistema inteligente de riego automatizado")
    st.caption("Lógica difusa (Mamdani) + optimización por algoritmo genético")

    tab_panel, tab_mora = st.tabs(["📊 Panel de Control", "🌱 Cultivo de Mora"])

    with tab_panel:
        col_izq, col_der = st.columns([1, 1.35])

        with col_izq:
            st.subheader("1. Entradas")
            temp = st.slider("Temperatura (°C)", 0.0, 50.0, key="slider_temp", step=0.5)
            hum = st.slider("Humedad del suelo (%)", 0.0, 100.0, key="slider_hum", step=1.0)
            st.session_state.temp = temp
            st.session_state.humedad = hum

            if st.button("🎲 Generar datos aleatorios", key="btn_rand"):
                st.session_state._pendiente_datos_aleatorios = generar_datos()
                st.rerun()

            st.subheader("2. Configuración")
            opts_tipo = ["Triangular", "Gaussiana"]
            idx_tipo = opts_tipo.index(st.session_state.tipo_ui) if st.session_state.tipo_ui in opts_tipo else 0
            tipo_ui = st.selectbox(
                "Tipo de función de membresía (categorías medias)",
                opts_tipo,
                index=idx_tipo,
            )
            st.session_state.tipo_ui = tipo_ui
            tipo_fn = _map_tipo(tipo_ui)
            st.caption(
                "Gaussiana: medias tipo campana (temperatura y humedad); extremos trapezoidales. "
                "Triangular: medias triangulares en temperatura y humedad."
            )

            n_gen = st.slider("Número de generaciones", 30, 100, int(st.session_state.n_gen), 1)
            n_pop = st.slider("Tamaño de población", 20, 50, int(st.session_state.n_pop), 1)
            st.session_state.n_gen = n_gen
            st.session_state.n_pop = n_pop

            st.subheader("3. Acciones")
            if st.button("💧 Ver resultado actual", type="primary"):
                st.session_state.ultimo_aviso = (
                    "Resultado actual mostrado (se actualiza también al mover los sliders)."
                )
            if st.button("🧬 Optimizar con Algoritmo Genético"):
                with st.spinner("Evolucionando población y ajustando membresías…"):
                    try:
                        comp = ejecutar_con_genetico(
                            temp,
                            hum,
                            tipo_funcion=tipo_fn,
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
                    st.success("Optimización completada.")
            if st.button("🔄 Reiniciar"):
                st.session_state._pendiente_reinicio = True
                st.rerun()

            if getattr(st.session_state, "ultimo_aviso", None):
                st.info(st.session_state.ultimo_aviso)

        tipo_fn = _map_tipo(st.session_state.tipo_ui)
        params_plot = st.session_state.params_grafico
        try:
            datos = curvas_membresia_para_grafico(tipo_funcion=tipo_fn, parametros=params_plot)
            live = calcular_riego(temp, hum, tipo_funcion=tipo_fn, parametros=None)
        except Exception as e:
            st.error(f"No se pudo evaluar el sistema difuso: {e}")
            return

        with col_der:
            st.subheader("4. Funciones de membresía")
            if params_plot is not None:
                st.caption(
                    "Curvas con parámetros optimizados por el AG (si aplica). Tras Reiniciar vuelven al diseño base + genes por defecto."
                )
            else:
                st.caption("Curvas con parámetros por defecto. Optimice para ver el ajuste evolutivo.")

            f1 = _fig_membresia(
                "Temperatura",
                datos["temp_x"],
                datos["temp"],
                temp,
                "Temperatura (°C)",
            )
            st.pyplot(f1, clear_figure=True)
            plt.close(f1)

            f2 = _fig_membresia(
                "Humedad del suelo",
                datos["humedad_x"],
                datos["humedad"],
                hum,
                "Humedad (%)",
            )
            st.pyplot(f2, clear_figure=True)
            plt.close(f2)

            f3 = _fig_riego_salida(datos["riego"], datos["riego_x"])
            st.pyplot(f3, clear_figure=True)
            plt.close(f3)

            st.subheader("5. Resultado en tiempo real (sin optimizar genes)")
            agua = live["agua"]
            st.metric("Agua recomendada", f"{agua:.2f} L")
            st.markdown(_nivel_color_html(live["nivel"]), unsafe_allow_html=True)
            st.progress(min(max(agua / 10.0, 0.0), 1.0))

            st.subheader("Reporte PDF")
            try:
                from logica.reporte_pdf import generar_reporte_sesion_pdf

                pdf_bytes = generar_reporte_sesion_pdf(
                    temperatura=float(temp),
                    humedad=float(hum),
                    tipo_ui=str(st.session_state.tipo_ui),
                    tipo_fn=tipo_fn,
                    n_generaciones=int(st.session_state.n_gen),
                    n_poblacion=int(st.session_state.n_pop),
                    resultado_sin=live,
                    parametros_grafico=st.session_state.params_grafico,
                    comparacion=st.session_state.comparacion,
                )
                nombre_pdf = f"reporte_riego_{datetime.now():%Y%m%d_%H%M%S}.pdf"
                st.download_button(
                    "📥 Descargar reporte PDF",
                    data=pdf_bytes,
                    file_name=nombre_pdf,
                    mime="application/pdf",
                    help="Incluye entradas, resultados difusos, reglas activadas, gráficas de membresía y evolución del fitness si hubo AG.",
                )
            except ModuleNotFoundError:
                st.warning(
                    "Para el PDF falta **reportlab** (y **Pillow**). En la terminal ejecuta: "
                    "`python -m pip install reportlab Pillow` — debe ser el mismo `python` que usas para Streamlit."
                )
            except Exception as ex:
                st.warning(f"No se pudo generar el PDF: {ex}")

            comp = st.session_state.comparacion
            if comp is not None:
                st.subheader("6. Comparación antes / después del AG")
                sin_o = comp["resultado_sin_optimizar"]
                con_o = comp["resultado_optimizado"]
                fit = comp["mejor_fitness"]
                tabla = f"""
| Métrica | Sin optimizar | Con genético |
| :--- | :---: | :---: |
| Agua recomendada (L) | {sin_o['agua']:.2f} | {con_o['agua']:.2f} |
| Nivel | {sin_o['nivel']} | {con_o['nivel']} |
| Fitness final | — | {fit:.4f} |
"""
                st.markdown(tabla)

                st.subheader("7. Evolución del fitness")
                hist = comp["historial"]
                fig_e, ax_e = plt.subplots(figsize=(6.5, 3.4), facecolor="white")
                ax_e.set_facecolor("#fafafa")
                ax_e.plot(range(1, len(hist) + 1), hist, color=COLOR_BAJA, linewidth=2, marker="o", markersize=3)
                ax_e.set_xlabel("Generación")
                ax_e.set_ylabel("Mejor fitness acumulado")
                ax_e.set_title("Evolución del mejor fitness (menor es mejor)", fontsize=12, fontweight="bold")
                ax_e.grid(True, alpha=0.3)
                fig_e.tight_layout()
                st.pyplot(fig_e, clear_figure=True)
                plt.close(fig_e)

            with st.expander("Activaciones de reglas (tiempo real)"):
                st.json(live["activaciones"])

    with tab_mora:
        render_cultivo_mora()
