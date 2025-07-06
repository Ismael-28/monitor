#!/usr/bin/env python3
# main.py
"""
Script principal para monitorizar la señal Wi-Fi (RSSI) y el Punto de Acceso (AP).

También permite la captura de paquetes con TShark en una interfaz en modo monitor.
"""

import os
from datetime import datetime
from typing import Optional
import typer
from utils import get_interface_display_name
from theme import console
import system_utils
import ui
from plotting_collector_2 import RealTimePlot, MATPLOTLIB_AVAILABLE

app = typer.Typer(add_completion=False)


def setup_logging(interface_name: str, name: Optional[str]) -> Optional[object]:
    """Configura y abre el archivo de log si es necesario."""
    base_dir = "logs"
    iface_dir = os.path.join(base_dir, get_interface_display_name(interface_name))
    os.makedirs(iface_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    suffix = name or ""

    log_filename = os.path.join(iface_dir, f"mon_{interface_name}_{suffix}_{timestamp}.log")
    try:
        log_file = open(log_filename, 'w', encoding='utf-8')
        console.print(f"Guardando log en: {log_filename}")
        return log_file
    except IOError as e:
        console.print(f"[error]Error al abrir el archivo de log: {e}[/error]")
        return None


@app.command()
def main(
    interface: str = typer.Argument(
        None,
        help='Interfaz Wi-Fi a monitorizar (ej: wlan0).'
    ),
    interval: float = typer.Option(
        1.0, '-i', '--interval',
        help='Intervalo en segundos (def: 1.0).'
    ),
    capture_interface: str = typer.Option(
        None, '-w', '--capture-interface',
        help='Interfaz para captura con TShark.'
    ),
    log: bool = typer.Option(
        False, '-l', '--log',
        help='Guardar la salida en un archivo de log.'
    ),
    target: str = typer.Option(
        None, '-t', '--target',
        help='IP o hostname para medir latencia, jitter y pérdida de paquetes.'
    ),
    name: str = typer.Option(
        None, '-n', '--name',
        help='Etiqueta a añadir antes del timestamp en los ficheros.'
    )
):
    """Función principal que orquesta la ejecución del script."""
    system_utils.check_dependencies()

    # Selección de interfaz de monitor (Managed)
    mon_iface = interface or ui.select_interface_dialog(mode='Managed')
    if not mon_iface:
        console.print("[error]No se seleccionó una interfaz de monitorización. Saliendo.[/error]")
        raise typer.Exit(code=1)

    # Selección de interfaz para captura (Monitor)
    capture_iface = (
        capture_interface
        if capture_interface is not None
        else ui.select_interface_dialog(mode='Monitor')
    )

    log_file = setup_logging(mon_iface, name) if log else None

    # Verificamos Matplotlib para decidir sobre target
    if MATPLOTLIB_AVAILABLE:
        target = target or ui.get_target_dialog()
        if not target:
            console.print("[error]El objetivo de ping es obligatorio para la gráfica. Saliendo.[/error]")
            raise typer.Exit(code=1)
    else:
        target = None

    # Iniciar captura de TShark si se pidió
    tshark_proc = pcap_path = None
    if capture_iface:
        tshark_proc, pcap_path = system_utils.start_tshark_capture(capture_iface, mon_iface, name)

    plotter = None
    try:
        if not MATPLOTLIB_AVAILABLE:
            console.print("[error]Matplotlib no disponible: no se puede mostrar la gráfica.[/error]")
            raise typer.Exit(code=1)

        plotter = RealTimePlot(mon_iface, interval, log_file, target, name)
        plotter.start()

    except (KeyboardInterrupt, SystemExit):
        console.print("\n\n[error]Programa detenido por el usuario.[/error]")

    finally:
        # Detener captura
        if tshark_proc:
            system_utils.stop_tshark_capture(tshark_proc)

        # Cerrar log
        if log_file:
            log_file.close()
            console.print("[success]Log cerrado.[/success]")

        # Mostrar estadísticas y preguntar por guardado
        ui.save_plot_dialog(plotter)

        # Confirmar guardado de pcap
        if pcap_path:
            ui.confirm_save("Captura de tshark", pcap_path)

        console.print("\n[success]Script finalizado.[/success]")


if __name__ == "__main__":
    app()
