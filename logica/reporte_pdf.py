"""
Genera un reporte de sesión en PDF (ReportLab + gráficas Matplotlib).
Sin dependencias de Streamlit: devuelve bytes para descarga.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .fuzzy_system import curvas_membresia_para_grafico

COLOR_BAJA = "#1f77b4"
COLOR_MEDIA = "#2ca02c"
COLOR_ALTA = "#d62728"
COLOR_ACENTO = "#555555"

_ETIQUETAS_REGLAS: Dict[str, str] = {
    "R1_humedad_baja_temp_alta_riego_alto": "R1 humedad baja + temp alta",
    "R2_humedad_baja_temp_media_riego_alto": "R2 humedad baja + temp media",
    "R3_humedad_baja_temp_baja_riego_medio": "R3 humedad baja + temp baja",
    "R4_humedad_media_temp_alta_riego_medio": "R4 humedad media + temp alta",
    "R5_humedad_media_temp_media_riego_medio": "R5 humedad media + temp media",
    "R6_humedad_media_temp_baja_riego_bajo": "R6 humedad media + temp baja",
    "R7_humedad_alta_temp_alta_riego_bajo": "R7 humedad alta + temp alta",
    "R8_humedad_alta_temp_media_riego_bajo": "R8 humedad alta + temp media",
    "R9_humedad_alta_temp_baja_riego_bajo": "R9 humedad alta + temp baja",
}


def _registrar_fuente_unicode() -> str:
    """Registra una TTF con soporte UTF-8 (Windows / Linux común)."""
    nombre = "ReporteFont"
    if nombre in pdfmetrics.getRegisteredFontNames():
        return nombre
    candidatos: List[str] = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        for fname in ("arial.ttf", "calibri.ttf", "segoeui.ttf"):
            candidatos.append(os.path.join(windir, "Fonts", fname))
    candidatos.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    )
    for path in candidatos:
        if path and os.path.isfile(path):
            pdfmetrics.registerFont(TTFont(nombre, path))
            return nombre
    return "Helvetica"


def _estilos(fuente: str) -> Tuple[Any, ParagraphStyle, ParagraphStyle, ParagraphStyle]:
    base = getSampleStyleSheet()
    titulo = ParagraphStyle(
        "TituloRep",
        parent=base["Heading1"],
        fontName=fuente,
        fontSize=16,
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    subt = ParagraphStyle(
        "SubtRep",
        parent=base["Heading2"],
        fontName=fuente,
        fontSize=12,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#333333"),
    )
    cuerpo = ParagraphStyle(
        "CuerpoRep",
        parent=base["Normal"],
        fontName=fuente,
        fontSize=10,
        leading=13,
    )
    return base, titulo, subt, cuerpo


def _fig_membresia(
    titulo: str,
    x: np.ndarray,
    curvas: Dict[str, np.ndarray],
    x_actual: Optional[float],
    xlabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.8), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(x, curvas["baja"], color=COLOR_BAJA, label="Baja", linewidth=1.8)
    ax.plot(x, curvas["media"], color=COLOR_MEDIA, label="Media", linewidth=1.8)
    ax.plot(x, curvas["alta"], color=COLOR_ALTA, label="Alta", linewidth=1.8)
    if x_actual is not None:
        ax.axvline(x_actual, color=COLOR_ACENTO, linestyle="--", linewidth=1.2, label="Actual")
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("μ", fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_riego(curvas: Dict[str, np.ndarray], x: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.8), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(x, curvas["bajo"], color=COLOR_BAJA, label="Bajo", linewidth=1.8)
    ax.plot(x, curvas["medio"], color=COLOR_MEDIA, label="Medio", linewidth=1.8)
    ax.plot(x, curvas["alto"], color=COLOR_ALTA, label="Alto", linewidth=1.8)
    ax.set_title("Riego (salida)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Litros", fontsize=9)
    ax.set_ylabel("μ", fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_fitness(hist: List[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.8), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(range(1, len(hist) + 1), hist, color=COLOR_BAJA, linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Generación", fontsize=9)
    ax.set_ylabel("Mejor fitness", fontsize=9)
    ax.set_title("Evolución del fitness (menor es mejor)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_a_imagen(fig: plt.Figure, dpi: int = 110) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def generar_reporte_sesion_pdf(
    temperatura: float,
    humedad: float,
    tipo_ui: str,
    tipo_fn: str,
    n_generaciones: int,
    n_poblacion: int,
    resultado_sin: Dict[str, Any],
    parametros_grafico: Optional[np.ndarray] = None,
    comparacion: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Construye el PDF de la sesión actual.

    resultado_sin: salida de calcular_riego (sin optimizar genes).
    comparacion: salida de ejecutar_con_genetico o None.
    """
    try:
        fuente = _registrar_fuente_unicode()
        _, est_titulo, est_subt, est_cuerpo = _estilos(fuente)

        buffer_pdf = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer_pdf,
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.2 * cm,
            bottomMargin=1.2 * cm,
        )
        story: List[Any] = []

        ahora = datetime.now().strftime("%d/%m/%Y %H:%M")
        titulo_doc = (
            "REPORTE DE SESIÓN — Sistema de Riego Inteligente"
            if fuente != "Helvetica"
            else "REPORTE DE SESION - Sistema de Riego Inteligente"
        )
        story.append(Paragraph(titulo_doc, est_titulo))
        story.append(Paragraph(f"Fecha y hora: {ahora}", est_cuerpo))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("-" * 56, est_cuerpo))

        story.append(Paragraph("DATOS DE ENTRADA", est_subt))
        datos_txt = f"""
<b>Temperatura:</b> {temperatura:.1f} °C<br/>
<b>Humedad:</b> {humedad:.1f} %<br/>
<b>Tipo función:</b> {tipo_ui}<br/>
<b>Generaciones (config):</b> {n_generaciones} &nbsp;|&nbsp; <b>Población:</b> {n_poblacion}
"""
        story.append(Paragraph(datos_txt.strip(), est_cuerpo))
        story.append(Spacer(1, 0.3 * cm))

        sin_agua = float(resultado_sin["agua"])
        sin_nivel = str(resultado_sin["nivel"]).upper()
        story.append(Paragraph("RESULTADO SISTEMA DIFUSO (sin optimizar)", est_subt))
        story.append(
            Paragraph(
                f"<b>Agua recomendada:</b> {sin_agua:.2f} L<br/><b>Nivel:</b> {sin_nivel}",
                est_cuerpo,
            )
        )
        story.append(Spacer(1, 0.25 * cm))

        if comparacion is not None:
            con = comparacion["resultado_optimizado"]
            fit = float(comparacion["mejor_fitness"])
            con_agua = float(con["agua"])
            con_nivel = str(con["nivel"]).upper()
            diff = con_agua - sin_agua
            pct = (diff / sin_agua * 100.0) if abs(sin_agua) > 1e-9 else 0.0
            story.append(Paragraph("RESULTADO TRAS OPTIMIZACIÓN GENÉTICA", est_subt))
            opt_txt = f"""
<b>Agua recomendada:</b> {con_agua:.2f} L<br/>
<b>Nivel:</b> {con_nivel}<br/>
<b>Fitness final:</b> {fit:.4f}<br/>
<b>Generaciones:</b> {n_generaciones} &nbsp;|&nbsp; <b>Población:</b> {n_poblacion}<br/>
<b>Mejora en agua:</b> {diff:+.2f} L ({pct:+.1f} %)
"""
            story.append(Paragraph(opt_txt.strip(), est_cuerpo))
        else:
            story.append(Paragraph("RESULTADO TRAS OPTIMIZACIÓN GENÉTICA", est_subt))
            story.append(
                Paragraph(
                    "<i>No se ha ejecutado la optimización genética en esta sesión.</i>",
                    est_cuerpo,
                )
            )
        story.append(Spacer(1, 0.35 * cm))

        story.append(Paragraph("REGLAS ACTIVADAS (grado de activación)", est_subt))
        act: Dict[str, float] = resultado_sin.get("activaciones", {})
        lineas_reglas: List[str] = []
        for clave in sorted(act.keys(), key=lambda k: k):
            etiqueta = _ETIQUETAS_REGLAS.get(clave, clave)
            lineas_reglas.append(f"{etiqueta} -&gt; {act[clave]:.4f}")
        story.append(Paragraph("<br/>".join(lineas_reglas), est_cuerpo))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph("GRÁFICAS — FUNCIONES DE MEMBRESÍA", est_subt))
        datos = curvas_membresia_para_grafico(tipo_funcion=tipo_fn, parametros=parametros_grafico)

        ancho_img = 16 * cm
        alto_img = 8 * cm

        for titulo, x_arr, curvas, x0, xlab in [
            ("Temperatura", datos["temp_x"], datos["temp"], temperatura, "Temperatura (°C)"),
            ("Humedad del suelo", datos["humedad_x"], datos["humedad"], humedad, "Humedad (%)"),
        ]:
            fig = _fig_membresia(titulo, x_arr, curvas, x0, xlab)
            buf = _fig_a_imagen(fig)
            story.append(Image(buf, width=ancho_img, height=alto_img))
            story.append(Spacer(1, 0.2 * cm))

        fig_r = _fig_riego(datos["riego"], datos["riego_x"])
        buf_r = _fig_a_imagen(fig_r)
        story.append(Image(buf_r, width=ancho_img, height=alto_img))
        story.append(Spacer(1, 0.35 * cm))

        story.append(Paragraph("EVOLUCIÓN DEL FITNESS", est_subt))
        if comparacion is not None and comparacion.get("historial"):
            fig_e = _fig_fitness(comparacion["historial"])
            buf_e = _fig_a_imagen(fig_e)
            story.append(Image(buf_e, width=ancho_img, height=alto_img))
        else:
            story.append(
                Paragraph(
                    "<i>No hay historial de fitness (ejecute primero el algoritmo genético).</i>",
                    est_cuerpo,
                )
            )

        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Resumen numérico", est_subt))
        if comparacion is not None:
            con = comparacion["resultado_optimizado"]
            tabla_datos = [
                ["Métrica", "Sin optimizar", "Con genético"],
                ["Agua (L)", f"{sin_agua:.2f}", f"{float(con['agua']):.2f}"],
                ["Nivel", sin_nivel.lower(), str(con["nivel"]).lower()],
                ["Fitness", "—", f"{float(comparacion['mejor_fitness']):.4f}"],
            ]
        else:
            tabla_datos = [
                ["Métrica", "Valor"],
                ["Agua recomendada (L)", f"{sin_agua:.2f}"],
                ["Nivel", sin_nivel.lower()],
            ]
        t = Table(tabla_datos, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                    ("FONTNAME", (0, 0), (-1, -1), fuente),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                ]
            )
        )
        story.append(t)

        doc.build(story)
        out = buffer_pdf.getvalue()
        buffer_pdf.close()
        return out
    except Exception as e:
        raise RuntimeError(f"No se pudo generar el reporte PDF: {e}") from e
