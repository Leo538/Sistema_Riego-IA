"""
Vista «Cultivo de Mora»: visualización viva ligada a temp/humedad del session_state
y a calcular_riego (misma lógica que el panel de control, sin duplicar sliders).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle

from logica.fuzzy.sistema import calcular_riego


def _map_tipo(ui_val: str) -> str:
    return "gaussiana" if ui_val == "Gaussiana" else "triangular"


def _fondo_temp_css(temp: float) -> str:
    if temp < 15:
        return "#cfe8ff"
    if temp <= 25:
        return "#d4edda"
    if temp <= 35:
        return "#fff9c4"
    return "#ffe0b2"


def _planta_escena(temp: float, humedad: float) -> str:
    if temp < 10:
        return """❄️🌿❄️<br/><span style="letter-spacing:0.15em;">║ ║ ║</span><br/>🧊🧊🧊"""
    if humedad > 85:
        return """🍄🌿🍄<br/><span style="letter-spacing:0.15em;">║ ║ ║</span><br/>💧💧💧"""
    if humedad < 30:
        return """🍂🥀🍂<br/><span style="letter-spacing:0.15em;">║ ║ ║</span><br/>🌾🌾🌾"""
    if 50 <= humedad <= 80 and 15 <= temp <= 25:
        return """🌿🍇🌿<br/><span style="letter-spacing:0.15em;">║ ║ ║</span><br/>🌱🌱🌱"""
    return """🌿🍇🌿<br/><span style="letter-spacing:0.15em;">║ ║ ║</span><br/>🌱🌱🌱"""


def _indicador_riego_html(agua: float) -> str:
    if agua < 2:
        return "🌂 Sin riego necesario"
    if agua <= 5:
        return "🌦️ Riego moderado 💧💧"
    return "🌧️ Riego intenso 💧💧💧💧💧"


def _etiqueta_temperatura(temp: float) -> Tuple[str, str]:
    if temp < 15:
        return "Fría", "#1565c0"
    if temp <= 25:
        return "Ideal", "#2e7d32"
    if temp <= 35:
        return "Cálida", "#ef6c00"
    return "Extrema", "#c62828"


def _etiqueta_humedad(h: float) -> Tuple[str, str]:
    if h < 30:
        return "Seco", "#c62828"
    if h < 50:
        return "Bajo", "#ef6c00"
    if h <= 80:
        return "Óptimo", "#2e7d32"
    return "Exceso", "#1565c0"


def _nivel_riego_colores(nivel: str) -> str:
    n = nivel.lower()
    if n == "bajo":
        return "#2e7d32"
    if n == "medio":
        return "#ef6c00"
    return "#c62828"


def _resolver_estado_cultivo(
    temp: float, humedad: float, agua: float
) -> Tuple[str, str, str, str]:
    """
    Devuelve (estado_html, color_hex, tipo_alerta, texto_largo).
    tipo_alerta: success | warning | error | info
    Prioridad: condiciones más críticas / encharcado primero.
    """
    t, h, a = temp, humedad, agua

    if h > 85:
        return (
            "🍄 Exceso de humedad",
            "#6a1b9a",
            "error",
            f"Con {h:.0f}% de humedad el suelo está encharcado. "
            f"El sistema recomienda {a:.2f}L mínimos pero la prioridad es el drenaje. "
            "Riesgo alto de pudrición de raíces.",
        )
    if t > 35 and h < 30:
        return (
            "🔥 Estrés hídrico severo",
            "#b71c1c",
            "error",
            f"Situación crítica: {t:.1f}°C con apenas {h:.0f}% de humedad. "
            f"La mora puede sufrir daño irreversible en hojas y frutos. "
            f"Riego inmediato de {a:.2f}L es urgente, preferiblemente por goteo en horas de menor calor.",
        )
    if t > 35 and h > 70:
        return (
            "😰 Calor con exceso de agua",
            "#4a148c",
            "error",
            "Alta temperatura combinada con exceso de humedad crea condiciones ideales "
            "para hongos como Botrytis. Suspender riego, mejorar drenaje y aplicar fungicida "
            "preventivo si es necesario.",
        )
    if t < 15 and h > 70:
        return (
            "🥶 Riesgo de helada",
            "#0d47a1",
            "error",
            f"Con {t:.1f}°C y {h:.0f}% de humedad, la mora enfrenta riesgo de daño por frío. "
            "El exceso de agua agrava la situación. Se recomienda suspender el riego y proteger "
            "el cultivo con malla antigranizo.",
        )
    if t < 15 and h < 40:
        return (
            "🌨️ Frío y seco",
            "#42a5f5",
            "warning",
            "Temperatura baja con suelo seco. La mora reduce su metabolismo en frío, "
            f"por lo que necesita {a:.2f}L mínimos para mantener hidratación sin encharcamiento.",
        )
    if 15 <= t <= 25 and 50 <= h <= 80:
        return (
            "🌿 Condiciones ideales",
            "#2e7d32",
            "success",
            "Temperatura y humedad en rango óptimo para mora. El sistema recomienda "
            f"{a:.2f}L para mantener las condiciones actuales. "
            "Este es el escenario ideal para producción de fruta de calidad.",
        )
    if 15 <= t <= 25 and h < 40:
        return (
            "💧 Suelo seco — regar ahora",
            "#f57c00",
            "warning",
            "La temperatura es ideal pero el suelo está por debajo del nivel óptimo. "
            f"Con {a:.2f}L de riego se puede alcanzar la humedad óptima para el cultivo de mora.",
        )
    if 25 < t <= 35 and h < 40:
        return (
            "⚠️ Estrés hídrico moderado",
            "#e65100",
            "warning",
            f"Con {t:.1f}°C y solo {h:.0f}% de humedad, la mora experimenta estrés hídrico. "
            f"La evapotranspiración es elevada. Se recomiendan {a:.2f}L de riego para compensar "
            "la pérdida de agua por calor.",
        )
    return (
        "🌱 Condiciones aceptables",
        "#66bb6a",
        "info",
        "Las condiciones actuales son aceptables para el cultivo de mora. El sistema recomienda "
        f"{a:.2f}L para mantener estabilidad. Monitorear cambios de temperatura.",
    )


def _mostrar_alerta(tipo: str, texto: str) -> None:
    if tipo == "success":
        st.success(texto)
    elif tipo == "warning":
        st.warning(texto)
    elif tipo == "error":
        st.error(texto)
    else:
        st.info(texto)


def _mapa_condiciones(temp: float, humedad: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), facecolor="white")
    ax.set_facecolor("#fafafa")
    # Estrés: todo el rectángulo
    ax.add_patch(Rectangle((0, 0), 50, 100, facecolor="#ffcccc", edgecolor="none", zorder=1, label="_stress"))
    # Aceptable: 10–35 °C, 30–85 %
    ax.add_patch(Rectangle((10, 30), 25, 55, facecolor="#fff3b0", edgecolor="none", zorder=2))
    # Óptima: 15–25 °C, 50–80 %
    ax.add_patch(Rectangle((15, 50), 10, 30, facecolor="#c8e6c9", edgecolor="none", zorder=3))
    ax.scatter(
        [temp],
        [humedad],
        s=280,
        c="red",
        edgecolors="black",
        linewidths=2,
        zorder=5,
        label="Posición actual",
    )
    ax.annotate(
        "← Ahora",
        xy=(temp, humedad),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        zorder=6,
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Humedad del suelo (%)")
    ax.set_title("Mapa de condiciones — Mora (Rubus glaucus)", fontsize=12, fontweight="bold")
    leyenda: List[Any] = [
        Rectangle((0, 0), 1, 1, facecolor="#c8e6c9", edgecolor="gray", label="Zona óptima"),
        Rectangle((0, 0), 1, 1, facecolor="#fff3b0", edgecolor="gray", label="Zona aceptable"),
        Rectangle((0, 0), 1, 1, facecolor="#ffcccc", edgecolor="gray", label="Zona de estrés"),
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=12,
            linestyle="None",
            label="Posición actual",
        ),
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def render_cultivo_mora() -> None:
    st.header("🌱 Cultivo de Mora (*Rubus glaucus*)")
    st.caption(
        "Visualización viva según temperatura y humedad del **Panel de Control**. "
        "Ajusta los sliders en la otra pestaña para actualizar esta vista."
    )

    temp = float(st.session_state.get("temp", 25.0))
    humedad = float(st.session_state.get("humedad", 50.0))
    tipo_fn = _map_tipo(str(st.session_state.get("tipo_ui", "Triangular")))

    try:
        res: Dict[str, Any] = calcular_riego(temp, humedad, tipo_funcion=tipo_fn, parametros=None)
    except Exception as e:
        st.error(f"No se pudo calcular el riego: {e}")
        return

    agua = float(res["agua"])
    nivel = str(res["nivel"])

    # —— SECCIÓN 1: escena visual ——
    bg = _fondo_temp_css(temp)
    planta = _planta_escena(temp, humedad)
    riego_txt = _indicador_riego_html(agua)

    escena = f"""
<style>
#mora-escena .valor-temp,
#mora-escena .valor-hum,
#mora-escena .valor-riego,
#mora-escena .indicador-riego {{
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
  opacity: 1 !important;
}}
</style>
<div id="mora-escena" style="
  min-height: 250px;
  border-radius: 16px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.12);
  background: linear-gradient(180deg, {bg} 0%, #ffffff 100%);
  border: 1px solid #e0e0e0;
  position: relative;
  padding: 16px 20px 24px 20px;
  font-family: system-ui, sans-serif;
  color: #000000 !important;
">
  <div style="text-align:center;font-size:1.05rem;margin-bottom:8px;font-weight:600;color:#000000 !important;">
    <span class="indicador-riego" style="font-weight:700;">{riego_txt}</span>
  </div>
  <div style="position:absolute;top:12px;left:16px;font-size:1rem;background:rgba(255,255,255,0.85);padding:6px 10px;border-radius:8px;">
    🌡️ <span class="valor-temp" style="font-weight:800;">{temp:.1f}°C</span>
  </div>
  <div style="position:absolute;top:12px;right:16px;font-size:1rem;background:rgba(255,255,255,0.85);padding:6px 10px;border-radius:8px;">
    💧 <span class="valor-hum" style="font-weight:800;">{humedad:.0f}%</span>
  </div>
  <div style="text-align:center;padding-top:36px;padding-bottom:8px;font-size:2.4rem;line-height:1.35;">
    {planta}
  </div>
  <div style="text-align:center;margin-top:12px;font-size:1.1rem;font-weight:700;">
    🚿 <span class="valor-riego" style="font-weight:800;">{agua:.2f} L recomendados</span>
  </div>
</div>
"""
    st.markdown(escena, unsafe_allow_html=True)

    # —— SECCIÓN 2: métricas ——
    col1, col2, col3 = st.columns(3)
    lab_t, col_t = _etiqueta_temperatura(temp)
    lab_h, col_h = _etiqueta_humedad(humedad)
    col_n = _nivel_riego_colores(nivel)

    with col1:
        st.markdown("**🌡️ Temperatura**")
        st.markdown(f'<p style="font-size:2rem;font-weight:800;margin:0;">{temp:.1f}°C</p>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:{col_t};font-weight:700;">{lab_t}</span>', unsafe_allow_html=True)

    with col2:
        st.markdown("**🌱 Humedad del suelo**")
        st.markdown(f'<p style="font-size:2rem;font-weight:800;margin:0;">{humedad:.0f}%</p>', unsafe_allow_html=True)
        st.progress(min(max(humedad / 100.0, 0.0), 1.0))
        st.markdown(f'<span style="color:{col_h};font-weight:700;">{lab_h}</span>', unsafe_allow_html=True)

    with col3:
        st.markdown("**🚿 Riego recomendado**")
        st.markdown(f'<p style="font-size:2rem;font-weight:800;margin:0;">{agua:.2f} L</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span style="color:{col_n};font-weight:800;font-size:1.2rem;">{nivel.upper()}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # —— SECCIÓN 3: narrativa ——
    st.subheader("Estado del cultivo")
    estado_txt, color_estado, tipo_alerta, texto_largo = _resolver_estado_cultivo(temp, humedad, agua)
    st.markdown(
        f'<p style="font-weight:700;font-size:1.15rem;color:{color_estado};margin-bottom:0.5rem;">{estado_txt}</p>',
        unsafe_allow_html=True,
    )
    _mostrar_alerta(tipo_alerta, texto_largo)

    st.divider()

    # —— SECCIÓN 4: mapa ——
    st.subheader("Mapa de condiciones")
    fig_m = _mapa_condiciones(temp, humedad)
    st.pyplot(fig_m, clear_figure=True)
    plt.close(fig_m)
