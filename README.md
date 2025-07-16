# Monitorización Wi‑Fi

Este proyecto proporciona herramientas para monitorizar en tiempo real la calidad de la conexión Wi‑Fi en Linux, recogiendo métricas como RSSI, latencia, jitter y pérdida de paquetes, y ofreciendo además captura de paquetes con TShark y generación de gráficas dinámicas y estáticas.

## Estructura de módulos

- **`config.py`**
  - Define las configuraciones y constantes globales del sistema:
    - `AP_MAP`: Mapa de puntos de acceso (BSSID) con nombres y colores.  
    - `INTERFACE_MAP`: Alias para nombres de interfaces.  
    - `PLOT_CONFIG`: Configuraciones de estilo (títulos, ejes, colores) para las gráficas.

- **`models.py`**
  - Contiene las clases de datos (`dataclasses`) usadas en toda la aplicación:
    - `Sample`: Representa una muestra con timestamp, métricas (RSSI, latencia, jitter, pérdida) y datos de AP.  
    - `PlotConfig`: Configuración de cada subgráfica de Matplotlib.  
    - `APChange`: Registra cambios de punto de acceso durante la sesión.

- **`data_collector.py`**
  - Implementa la lógica para recolectar métricas de red de forma concurrente:
    - `BaseCollector`: Clase abstracta base para collectors basados en hilos.  
    - `RSSICollector`, `LatencyCollector`, `Iperf3Collector`: Subclases que ejecutan comandos del sistema (`iwconfig`, `ping`, `iperf3`) y parsean su salida.  
    - `DataCollector`: Orquesta estos collectors, unifica muestras en objetos `Sample`, detecta cambios de AP, imprime logs y gestiona resumen final.

- **`plotting.py`**
  - Define funciones y clases para generar gráficas de las métricas:
    - Helpers (`style_axis`, `_setup_plot_axes`, `_draw_ap_change_annotations`).  
    - `generate_final_plot()`: Crea una figura estática con todos los datos recogidos.  
    - `RealTimePlot`: Clase que arranca la monitorización, actualiza gráficas en tiempo real vía `matplotlib.animation` y ofrece métodos para guardar imagen y CSV.

- **`system_utils.py`**
  - Funciones para interactuar con el sistema operativo:
    - `check_dependencies()`: Verifica que las herramientas externas (e.g. `iwconfig`, `tshark`) estén instaladas.  
    - `get_wifi_interfaces_list()`: Lista interfaces Wi‑Fi disponibles, opcionalmente filtradas por modo (Managed/Monitor).  
    - `start_tshark_capture()`, `stop_tshark_capture()`: Inician y detienen captura con TShark.

- **`theme.py`**
  - Configura el tema de colores para la consola usando `rich`:
    - Define colores semánticos y estilo de salida.  
    - Exporta la instancia `console` con ese tema.

- **`utils.py`**
  - Funciones auxiliares de uso común:
    - `get_interface_display_name()`: Devuelve un nombre amigable para una interfaz.  
    - `format_stat()`, `build_monitor_output()`: Formatean valores numéricos y líneas de salida.  
    - `write_log_line()`: Escribe cada muestra en un archivo de log CSV.

- **`ui.py`**
  - Maneja la interacción con el usuario vía consola:
    - `select_interface_dialog()`: Diálogo para seleccionar interfaz Wi‑Fi.  
    - `get_target_dialog()`: Solicita la IP/hostname de destino para ping.  
    - `confirm_save()`, `save_plot_dialog()`: Preguntas para guardar o descartar archivos generados.

- **`main.py`**
  - Punto de entrada de la aplicación (CLI con `Typer`):
    1. Verifica dependencias.  
    2. Selecciona interfaces Managed y Monitor.  
    3. Sincroniza canales entre interfaces.  
    4. Configura logging y captura TShark.  
    5. Inicializa y arranca `RealTimePlot`.  
    6. Al finalizar, detiene captura, cierra logs y ofrece guardar gráficas y datos.

## Requisitos

- **Python 3.7+**  
- Módulos Python:
  - `typer`, `rich`, `pandas`, `matplotlib`  
- Herramientas de sistema:
  - `iwconfig` (wireless-tools)  
  - `tshark` (Wireshark)  
  - `ping` (iputils)  
  - `iperf3`

## Uso

```bash
# Ejecutar monitorización interactiva
python3 main.py wlan0 -i 1.0 -l -t 8.8.8.8
```

Para más detalles, consulta los docstrings de cada módulo y ejecuta:

```bash
python3 main.py --help
```
