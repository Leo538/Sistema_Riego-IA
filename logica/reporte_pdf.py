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

from .fuzzy.membresia import VARIABLES_DIFUSAS
from .fuzzy.sistema import curvas_membresia_para_grafico

COLOR_BAJA = "#1f77b4"
COLOR_MEDIA = "#2ca02c"
COLOR_ALTA = "#d62728"
COLOR_ACENTO = "#555555"

_GENES_POR_VARIABLE = 11

_ETIQUETA_GRAFICO: Dict[str, Tuple[str, str]] = {
    "humedad_suelo": ("Humedad del suelo", "Humedad suelo (0-64)"),
    "temperatura": ("Temperatura", "Temperatura (C)"),
    "humedad_relativa": ("Humedad relativa", "Humedad relativa (%)"),
    "par": ("PAR", "PAR (0-2000)"),
}


def _registrar_fuente_unicode() -> str:
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
    ax.set_ylabel("mu", fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_riego(curvas: Dict[str, np.ndarray], x: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.8), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(x, curvas["bajo"], color=COLOR_BAJA, label="Corta", linewidth=1.8)
    ax.plot(x, curvas["medio"], color=COLOR_MEDIA, label="Media", linewidth=1.8)
    ax.plot(x, curvas["alto"], color=COLOR_ALTA, label="Larga", linewidth=1.8)
    ax.set_title("Duracion de riego (salida)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Minutos (0-30)", fontsize=9)
    ax.set_ylabel("mu", fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_fitness(hist: List[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.8), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(range(1, len(hist) + 1), hist, color=COLOR_BAJA, linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Generacion", fontsize=9)
    ax.set_ylabel("Mejor fitness", fontsize=9)
    ax.set_title("Evolucion del fitness (menor es mejor)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _genes_ag_str(chrom: np.ndarray, start: int, end: int) -> str:
    sl = np.asarray(chrom, dtype=float).flatten()[start:end]
    return str([round(float(x), 2) for x in sl])


def _ag_pdf_variable(parametros_grafico: Optional[np.ndarray], indice_variable: int) -> Tuple[str, str, str]:
    if parametros_grafico is None:
        return "—", "—", "—"
    p = np.asarray(parametros_grafico, dtype=float).flatten()
    s = indice_variable * _GENES_POR_VARIABLE
    bloque = p[s : s + _GENES_POR_VARIABLE]
    ag_baja = _genes_ag_str(bloque, 0, 4)
    ag_media = _genes_ag_str(bloque, 4, 7)
    ag_alta = _genes_ag_str(bloque, 7, 11)
    return ag_baja, ag_media, ag_alta


def _fmt_tpl(tpl: tuple[float, ...]) -> str:
    return str([round(float(x), 2) for x in tpl])


def _tabla_membresia_pdf(
    fuente: str,
    filas: List[List[str]],
) -> Table:
    datos = [
        ["Categoria", "Forma", "Parametros originales", "Parametros AG"],
    ] + filas
    t = Table(datos, hAlign="LEFT", colWidths=[3.2 * cm, 3.0 * cm, 5.2 * cm, 5.6 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("FONTNAME", (0, 0), (-1, -1), fuente),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def _nota_pdf_membresia_entrada(optimizado: bool, fuente: str, est_cuerpo: ParagraphStyle) -> Paragraph:
    txt = "AG ajusto estos parametros (11 genes por entrada)" if optimizado else "Ejecute el AG para ver la comparacion"
    if fuente != "Helvetica":
        pref = "[OK] " if optimizado else "[i] "
    else:
        pref = "[OK] " if optimizado else "[i] "
    return Paragraph(f"{pref}{txt}", est_cuerpo)


def _nota_pdf_riego(fuente: str, est_cuerpo: ParagraphStyle) -> Paragraph:
    msg = "La salida no es optimizada por el AG"
    pref = "[i] "
    return Paragraph(f"{pref}{msg}", est_cuerpo)


def _fig_a_imagen(fig: plt.Figure, dpi: int = 110) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _lineas_activaciones_pdf(activaciones: Any, max_filas: int = 24) -> List[str]:
    if not isinstance(activaciones, dict):
        return ["(sin datos de activacion)"]
    filas: List[Tuple[float, str]] = []
    for clave, val in activaciones.items():
        if isinstance(val, dict):
            g = float(val.get("grado", 0.0))
            desc = str(val.get("descripcion", "")).strip()
            texto = f"{clave}: mu={g:.4f}"
            if desc:
                texto += f" — {desc}"
            filas.append((g, texto))
        else:
            filas.append((0.0, f"{clave}: {val}"))
    filas.sort(key=lambda t: t[0], reverse=True)
    return [t[1] for t in filas[:max_filas]]


def generar_reporte_sesion_pdf(
    humedad_suelo: float,
    temperatura: float,
    humedad_relativa: float,
    par: float,
    n_generaciones: int,
    n_poblacion: int,
    resultado_sin: Dict[str, Any],
    parametros_grafico: Optional[np.ndarray] = None,
    comparacion: Optional[Dict[str, Any]] = None,
) -> bytes:
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
        titulo_doc = "REPORTE DE SESION - Sistema de Riego Inteligente"
        story.append(Paragraph(titulo_doc, est_titulo))
        story.append(Paragraph(f"Fecha y hora: {ahora}", est_cuerpo))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("-" * 56, est_cuerpo))

        story.append(Paragraph("DATOS DE ENTRADA", est_subt))
        datos_txt = f"""
<b>Humedad del suelo (0-64):</b> {humedad_suelo:.1f}<br/>
<b>Temperatura (C, 0-50):</b> {temperatura:.1f}<br/>
<b>Humedad relativa (%):</b> {humedad_relativa:.1f}<br/>
<b>PAR (0-2000):</b> {par:.1f}<br/>
<b>Membresias (entradas y salida):</b> trapecios en baja/alta, triangulos en media<br/>
<b>Generaciones (config):</b> {n_generaciones} &nbsp;|&nbsp; <b>Poblacion:</b> {n_poblacion}
"""
        story.append(Paragraph(datos_txt.strip(), est_cuerpo))
        story.append(Spacer(1, 0.3 * cm))

        sin_dur = float(resultado_sin["agua"])
        sin_nivel = str(resultado_sin["nivel"]).upper()
        story.append(Paragraph("RESULTADO SISTEMA DIFUSO (sin optimizar genes)", est_subt))
        story.append(
            Paragraph(
                f"<b>Duracion recomendada:</b> {sin_dur:.2f} min<br/><b>Nivel (UI):</b> {sin_nivel}",
                est_cuerpo,
            )
        )
        story.append(Spacer(1, 0.25 * cm))

        if comparacion is not None:
            con = comparacion["resultado_optimizado"]
            fit = float(comparacion["mejor_fitness"])
            con_dur = float(con["agua"])
            con_nivel = str(con["nivel"]).upper()
            diff = con_dur - sin_dur
            pct = (diff / sin_dur * 100.0) if abs(sin_dur) > 1e-9 else 0.0
            story.append(Paragraph("RESULTADO TRAS OPTIMIZACION GENETICA", est_subt))
            opt_txt = f"""
<b>Duracion recomendada:</b> {con_dur:.2f} min<br/>
<b>Nivel:</b> {con_nivel}<br/>
<b>Fitness final:</b> {fit:.4f}<br/>
<b>Generaciones:</b> {n_generaciones} &nbsp;|&nbsp; <b>Poblacion:</b> {n_poblacion}<br/>
<b>Cambio en duracion:</b> {diff:+.2f} min ({pct:+.1f} %)
"""
            story.append(Paragraph(opt_txt.strip(), est_cuerpo))
        else:
            story.append(Paragraph("RESULTADO TRAS OPTIMIZACION GENETICA", est_subt))
            story.append(
                Paragraph(
                    "<i>No se ha ejecutado la optimizacion genetica en esta sesion.</i>",
                    est_cuerpo,
                )
            )
        story.append(Spacer(1, 0.35 * cm))

        story.append(Paragraph("REGLAS ACTIVADAS (mayor grado primero, muestra max. 24)", est_subt))
        act = resultado_sin.get("activaciones", {})
        lineas_reglas = _lineas_activaciones_pdf(act)
        story.append(Paragraph("<br/>".join(lineas_reglas), est_cuerpo))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph("GRAFICAS - FUNCIONES DE MEMBRESIA", est_subt))
        datos = curvas_membresia_para_grafico(parametros=parametros_grafico)

        ancho_img = 16 * cm
        alto_img = 8 * cm

        optimizado = parametros_grafico is not None

        curvas_datos = [
            ("humedad_suelo", datos["humedad_x"], datos["humedad"], humedad_suelo),
            ("temperatura", datos["temp_x"], datos["temp"], temperatura),
            ("humedad_relativa", datos["humedad_rel_x"], datos["humedad_rel"], humedad_relativa),
            ("par", datos["par_x"], datos["par"], par),
        ]

        for idx, (clave, x_arr, curvas, x0) in enumerate(curvas_datos):
            titulo_graf, xlab = _ETIQUETA_GRAFICO[clave]
            fig = _fig_membresia(titulo_graf, x_arr, curvas, x0, xlab)
            buf = _fig_a_imagen(fig)
            story.append(Image(buf, width=ancho_img, height=alto_img))
            story.append(Spacer(1, 0.15 * cm))

            definicion = VARIABLES_DIFUSAS[clave]
            et = definicion.etiquetas
            ag_b, ag_m, ag_a = _ag_pdf_variable(parametros_grafico, idx)
            filas_mem = [
                ["Baja", "Trapezoidal", _fmt_tpl(tuple(et["baja"])), ag_b],
                ["Media", "Triangular", _fmt_tpl(tuple(et["media"])), ag_m],
                ["Alta", "Trapezoidal", _fmt_tpl(tuple(et["alta"])), ag_a],
            ]
            story.append(_tabla_membresia_pdf(fuente, filas_mem))
            story.append(Spacer(1, 0.12 * cm))
            story.append(_nota_pdf_membresia_entrada(optimizado, fuente, est_cuerpo))
            story.append(Spacer(1, 0.28 * cm))

        fig_r = _fig_riego(datos["riego"], datos["riego_x"])
        buf_r = _fig_a_imagen(fig_r)
        story.append(Image(buf_r, width=ancho_img, height=alto_img))
        story.append(Spacer(1, 0.15 * cm))
        et_r = VARIABLES_DIFUSAS["duracion_riego"].etiquetas
        if parametros_grafico is None:
            ag_rb, ag_rm, ag_ra = "—", "—", "—"
        else:
            ag_rb = ag_rm = ag_ra = "No optimizado"
        filas_riego = [
            ["Corta", "Trapezoidal", _fmt_tpl(tuple(et_r["corta"])), ag_rb],
            ["Media", "Triangular", _fmt_tpl(tuple(et_r["media"])), ag_rm],
            ["Larga", "Trapezoidal", _fmt_tpl(tuple(et_r["larga"])), ag_ra],
        ]
        story.append(_tabla_membresia_pdf(fuente, filas_riego))
        story.append(Spacer(1, 0.12 * cm))
        story.append(_nota_pdf_riego(fuente, est_cuerpo))
        story.append(Spacer(1, 0.28 * cm))

        story.append(Paragraph("EVOLUCION DEL FITNESS", est_subt))
        if comparacion is not None and comparacion.get("historial"):
            fig_e = _fig_fitness(comparacion["historial"])
            buf_e = _fig_a_imagen(fig_e)
            story.append(Image(buf_e, width=ancho_img, height=alto_img))
        else:
            story.append(
                Paragraph(
                    "<i>No hay historial de fitness (ejecute primero el algoritmo genetico).</i>",
                    est_cuerpo,
                )
            )

        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Resumen numerico", est_subt))
        if comparacion is not None:
            con = comparacion["resultado_optimizado"]
            tabla_datos = [
                ["Metrica", "Sin optimizar", "Con genetico"],
                ["Duracion (min)", f"{sin_dur:.2f}", f"{float(con['agua']):.2f}"],
                ["Nivel", sin_nivel.lower(), str(con["nivel"]).lower()],
                ["Fitness", "—", f"{float(comparacion['mejor_fitness']):.4f}"],
            ]
        else:
            tabla_datos = [
                ["Metrica", "Valor"],
                ["Duracion (min)", f"{sin_dur:.2f}"],
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
