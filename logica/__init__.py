# Paquete de lógica del sistema de riego inteligente.
from .fuzzy.sistema import calcular_riego, curvas_membresia_para_grafico
from .genetico.algoritmo import ejecutar_genetico
from .simulacion import generar_datos, ejecutar_sistema, ejecutar_con_genetico

# reporte_pdf no se importa aquí para no exigir reportlab al cargar el paquete.
# Usar: from logica.reporte_pdf import generar_reporte_sesion_pdf

__all__ = [
    "calcular_riego",
    "curvas_membresia_para_grafico",
    "ejecutar_genetico",
    "generar_datos",
    "ejecutar_sistema",
    "ejecutar_con_genetico",
]
