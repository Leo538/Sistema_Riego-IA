from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle

from logica.fuzzy.sistema import calcular_riego


def _fondo_temp_css(temp: float) -> str:
    if temp < 15:
        return "#cfe8ff"
    if temp <= 25:
        return "#d4edda"
    if temp <= 35:
        return "#fff9c4"
    return "#ffe0b2"


def _planta_escena(temp: float, humedad_suelo: float) -> str:
    if temp < 10:
        return "[Frio]  *** [Hielo]"
    if humedad_suelo > 56:
        return "[Suelo muy humedo]  ***  [Agua]"
    if humedad_suelo < 18:
        return "[Suelo seco]  ***  [Estres]"
    if 36 <= humedad_suelo <= 52 and 15 <= temp <= 25:
        return "[Cultivo estable]  ***  [OK]"
    return "[Cultivo]  ***  [Normal]"


def _indicador_riego_html(duracion: float) -> str:
    if duracion < 6:
        return "Riego corto (baja demanda)"
    if duracion <= 18:
        return "Riego moderado"
    return "Riego prolongado (alta demanda)"


def _etiqueta_temperatura(temp: float) -> Tuple[str, str]:
    if temp < 15:
        return "Fría", "#1565c0"
    if temp <= 25:
        return "Ideal", "#2e7d32"
    if temp <= 35:
        return "Cálida", "#ef6c00"
    return "Extrema", "#c62828"


def _etiqueta_humedad_suelo(h: float) -> Tuple[str, str]:
    if h < 18:
        return "Seco", "#c62828"
    if h < 30:
        return "Bajo", "#ef6c00"
    if h <= 52:
        return "Óptimo", "#2e7d32"
    return "Muy húmedo", "#1565c0"


def _nivel_riego_colores(nivel: str) -> str:
    n = nivel.lower()
    if n == "bajo":
        return "#2e7d32"
    if n == "medio":
        return "#ef6c00"
    return "#c62828"


def _resolver_estado_cultivo(
    temp: float,
    humedad_suelo: float,
    duracion: float,
    hr: float,
    par_v: float,
) -> Tuple[str, str, str, str]:
    t, hs, d = temp, humedad_suelo, duracion

    if hs > 56:
        return (
            "Exceso de humedad en suelo",
            "#6a1b9a",
            "error",
            f"Con {hs:.0f} (escala del modelo 0–64) el suelo está muy saturado. "
            f"El sistema sugiere una duración de {d:.1f} min; priorice drenaje y evite encharcar. "
            f"Humedad relativa {hr:.0f} %, PAR {par_v:.0f}.",
        )
    if t > 35 and hs < 22:
        return (
            "Estrés hídrico severo",
            "#b71c1c",
            "error",
            f"Situación crítica: {t:.1f} °C con suelo seco ({hs:.0f} en escala 0–64). "
            f"Se recomienda regar (duración sugerida {d:.1f} min) preferiblemente en horas frescas. "
            f"PAR alto ({par_v:.0f}) aumenta la demanda de agua.",
        )
    if t > 35 and hs > 48:
        return (
            "Calor con suelo húmedo",
            "#4a148c",
            "warning",
            "Alta temperatura con bastante agua en suelo: vigilar enfermedades fúngicas y ventilación. "
            f"Humedad relativa {hr:.0f} %.",
        )
    if t < 15 and hs > 48:
        return (
            "Frío y suelo húmedo",
            "#0d47a1",
            "warning",
            f"Con {t:.1f} °C y suelo húmedo ({hs:.0f}), reduzca riego si no hay evaporación. "
            "Riesgo de estrés por frío combinado con exceso de agua.",
        )
    if t < 15 and hs < 28:
        return (
            "Frío y suelo seco",
            "#42a5f5",
            "warning",
            "Temperatura baja con suelo por debajo del rango cómodo. "
            f"Riego mínimo ({d:.1f} min sugeridos) para evitar sequía sin encharcar.",
        )
    if 15 <= t <= 25 and 36 <= hs <= 52:
        return (
            "Condiciones favorables",
            "#2e7d32",
            "success",
            "Temperatura y humedad de suelo en rango favorable para el cultivo. "
            f"Duración sugerida {d:.1f} min. HR {hr:.0f} %, PAR {par_v:.0f}.",
        )
    if 15 <= t <= 25 and hs < 30:
        return (
            "Suelo seco: conviene regar",
            "#f57c00",
            "warning",
            "La temperatura es adecuada pero el suelo está bajo el nivel cómodo. "
            f"El difuso sugiere {d:.1f} min de riego.",
        )
    if 25 < t <= 35 and hs < 30:
        return (
            "Calor con suelo seco",
            "#e65100",
            "warning",
            f"Con {t:.1f} °C y poca humedad en suelo ({hs:.0f}), la evapotranspiración es alta. "
            f"Riego sugerido {d:.1f} min.",
        )
    return (
        "Condiciones aceptables",
        "#66bb6a",
        "info",
        "Condiciones aceptables para el cultivo. "
        f"Duración sugerida {d:.1f} min. Ajuste según observación en campo.",
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


def _mapa_condiciones(temp: float, humedad_suelo: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.add_patch(Rectangle((0, 0), 50, 64, facecolor="#ffcccc", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((10, 18), 25, 38, facecolor="#fff3b0", edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((15, 36), 10, 16, facecolor="#c8e6c9", edgecolor="none", zorder=3))
    ax.scatter(
        [temp],
        [humedad_suelo],
        s=280,
        c="red",
        edgecolors="black",
        linewidths=2,
        zorder=5,
        label="Posicion actual",
    )
    ax.annotate(
        "Ahora",
        xy=(temp, humedad_suelo),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        zorder=6,
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 64)
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Humedad del suelo (escala modelo 0–64)")
    ax.set_title("Mapa simplificado: suelo vs temperatura", fontsize=12, fontweight="bold")
    leyenda: List[Any] = [
        Rectangle((0, 0), 1, 1, facecolor="#c8e6c9", edgecolor="gray", label="Zona favorable"),
        Rectangle((0, 0), 1, 1, facecolor="#fff3b0", edgecolor="gray", label="Zona aceptable"),
        Rectangle((0, 0), 1, 1, facecolor="#ffcccc", edgecolor="gray", label="Zona de estres"),
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
            label="Posicion actual",
        ),
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def render_cultivo() -> None:
    st.header("Cultivo")
    st.caption(
        "Vista según las cuatro entradas del panel (suelo, temperatura, humedad relativa, PAR). "
        "Ajusta los controles en la pestaña «Panel de Control»."
    )

    hs = float(st.session_state.get("humedad_suelo", 36.0))
    temp = float(st.session_state.get("temp", 25.0))
    hr = float(st.session_state.get("humedad_relativa", 60.0))
    par_v = float(st.session_state.get("par", 1000.0))

    try:
        res: Dict[str, Any] = calcular_riego(
            temp,
            hs,
            parametros=None,
            humedad_relativa=hr,
            par=par_v,
        )
    except Exception as e:
        st.error(f"No se pudo calcular el riego: {e}")
        return

    duracion = float(res["agua"])
    nivel = str(res["nivel"])

    bg = _fondo_temp_css(temp)
    planta = _planta_escena(temp, hs)
    riego_txt = _indicador_riego_html(duracion)

    escena = f"""
<div style="
  min-height: 220px;
  border-radius: 16px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.12);
  background: linear-gradient(180deg, {bg} 0%, #ffffff 100%);
  border: 1px solid #e0e0e0;
  position: relative;
  padding: 16px 20px 24px 20px;
  font-family: system-ui, sans-serif;
  color: #000000;
">
  <div style="text-align:center;font-size:1.05rem;margin-bottom:8px;font-weight:600;">
    {riego_txt}
  </div>
  <div style="position:absolute;top:12px;left:16px;font-size:1rem;background:rgba(255,255,255,0.85);padding:6px 10px;border-radius:8px;">
    Temp: <b>{temp:.1f} °C</b>
  </div>
  <div style="position:absolute;top:12px;right:16px;font-size:1rem;background:rgba(255,255,255,0.85);padding:6px 10px;border-radius:8px;">
    Suelo: <b>{hs:.0f}</b> <span style="font-size:0.85rem;">(0–64)</span>
  </div>
  <div style="text-align:center;padding-top:28px;font-size:2.75rem;line-height:1;user-select:none;" title="Cultivo" aria-hidden="true">🌱</div>
  <div style="text-align:center;padding-top:6px;padding-bottom:8px;font-size:1.25rem;line-height:1.5;">
    {planta}
  </div>
  <div style="text-align:center;margin-top:12px;font-size:1.1rem;font-weight:700;">
    Duración sugerida: <b>{duracion:.1f} min</b>
  </div>
</div>
"""
    st.markdown(escena, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    lab_t, col_t = _etiqueta_temperatura(temp)
    lab_hs, col_hs = _etiqueta_humedad_suelo(hs)
    col_n = _nivel_riego_colores(nivel)

    with c1:
        st.markdown("**Temperatura**")
        st.markdown(f'<p style="font-size:1.6rem;font-weight:800;margin:0;">{temp:.1f} °C</p>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:{col_t};font-weight:700;">{lab_t}</span>', unsafe_allow_html=True)

    with c2:
        st.markdown("**Humedad suelo**")
        st.markdown(f'<p style="font-size:1.6rem;font-weight:800;margin:0;">{hs:.0f}</p>', unsafe_allow_html=True)
        st.progress(min(max(hs / 64.0, 0.0), 1.0))
        st.markdown(f'<span style="color:{col_hs};font-weight:700;">{lab_hs}</span>', unsafe_allow_html=True)
        st.caption("Rango modelo 0–64")

    with c3:
        st.markdown("**Hum. relativa**")
        st.markdown(f'<p style="font-size:1.6rem;font-weight:800;margin:0;">{hr:.0f}%</p>', unsafe_allow_html=True)
        st.progress(min(max(hr / 100.0, 0.0), 1.0))

    with c4:
        st.markdown("**PAR**")
        st.markdown(f'<p style="font-size:1.6rem;font-weight:800;margin:0;">{par_v:.0f}</p>', unsafe_allow_html=True)
        st.progress(min(max(par_v / 2000.0, 0.0), 1.0))
        st.caption("µmol·m⁻²·s⁻¹ · rango 0–2000")

    st.markdown(
        f'<p style="margin-top:12px;"><b>Duración difusa</b>: '
        f'<span style="color:{col_n};font-weight:800;">{duracion:.1f} min ({nivel.upper()})</span></p>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Estado del cultivo")
    estado_txt, color_estado, tipo_alerta, texto_largo = _resolver_estado_cultivo(
        temp, hs, duracion, hr, par_v
    )
    st.markdown(
        f'<p style="font-weight:700;font-size:1.15rem;color:{color_estado};margin-bottom:0.5rem;">{estado_txt}</p>',
        unsafe_allow_html=True,
    )
    _mostrar_alerta(tipo_alerta, texto_largo)

    st.divider()

    st.subheader("Mapa de condiciones (temperatura vs suelo)")
    fig_m = _mapa_condiciones(temp, hs)
    st.pyplot(fig_m, clear_figure=True)
    plt.close(fig_m)
