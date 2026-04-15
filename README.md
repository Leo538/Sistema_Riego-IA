# Sistema inteligente de riego automatizado

Aplicación con **lógica difusa (Mamdani)** y **algoritmo genético** para recomendar la duración del riego según **cuatro entradas** (humedad del suelo, temperatura, humedad relativa, PAR). Las membresías usan **trapecios** en etiquetas baja/alta y **triángulos** en la media. Interfaz web con **Streamlit**.

## Requisitos

- **Python 3.10 o superior** (recomendado 3.11+)
- `pip` actualizado

## Instalación

1. Entra en la carpeta del proyecto (si la ruta tiene espacios, usa comillas en la terminal):

   ```bash
   cd "ruta\al\proyecto\Riego Inteligente"
   ```

2. *(Opcional)* Crea y activa un entorno virtual:

   **Windows (PowerShell)**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Paquetes principales: `streamlit`, `numpy`, `scikit-fuzzy`, `matplotlib`, `networkx`, `reportlab`, `Pillow`.

## Cómo iniciar la aplicación

Desde la **raíz del proyecto** (donde está `main.py`):

```bash
python -m streamlit run main.py
```

Se abrirá el navegador (por defecto `http://localhost:8501`). Para detener el servidor, usa `Ctrl+C` en la terminal.

## Uso rápido

- Pestaña **Panel de Control**: ajusta las cuatro entradas, ejecuta el algoritmo genético si quieres optimizar las membresías de entrada y descarga el **PDF** del reporte (requiere `reportlab` y `Pillow`, ya listados en `requirements.txt`).
- Pestaña **Cultivo**: visualización ligada a los mismos valores del panel.

## Estructura del proyecto

| Ruta        | Descripción                                      |
| ----------- | ------------------------------------------------ |
| `main.py`   | Punto de entrada; lanza la interfaz Streamlit    |
| `ui/`       | Interfaz (`interfaz.py`, `cultivo.py`)             |
| `logica/`   | Difuso, AG, simulación, PDF                      |
| `requirements.txt` | Dependencias de Python                    |

## Notas

- Si `pip install` falla, comprueba que usas el mismo `python` / `pip` con el que ejecutarás Streamlit.
- No subas secretos: si creas `.streamlit/secrets.toml`, está ignorado en `.gitignore`.
