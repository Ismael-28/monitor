# system_utils.py
"""
Utilidades del sistema.

Funciones para interactuar con el sistema operativo, como verificar
dependencias y gestionar procesos externos (tshark).
"""

import subprocess
import re
import sys
import os
import shutil
from datetime import datetime
from typing import List, Optional, Tuple
from rich.console import Console

from utils import get_interface_display_name

console = Console()

def check_dependencies() -> None:
    """Verifica si los comandos listados están disponibles en el PATH."""
    console.print("[bold]Verificando dependencias...[/bold]")

    dependencies = [
        {
            "cmd": "iwconfig",
            "name": "iwconfig",
            "install_hint": "paquete 'wireless-tools'"
        },
        {
            "cmd": "tshark",
            "name": "tshark",
            "install_hint": "paquete 'wireshark' (normalmente incluye 'tshark')"
        },
    ]

    missing = []
    for dep in dependencies:
        if shutil.which(dep["cmd"]) is None:
            missing.append(dep)

    if missing:
        for dep in missing:
            console.print(f"[bold red]Error: El comando '{dep['name']}' no se encuentra en el PATH.[/bold red]")
            console.print(f"Este script lo necesita. Por favor, instálalo ({dep['install_hint']}).\n")
        sys.exit(1)

    nombres = ", ".join([dep["name"] for dep in dependencies])
    console.print(f"[bold green]Dependencias {nombres} encontradas.[/bold green]")


def get_wifi_interfaces_list(mode_filter: Optional[str] = None) -> List[str]:
    """Obtiene una lista de interfaces Wi-Fi, opcionalmente filtradas por modo."""
    try:
        result = subprocess.run(['iwconfig'], capture_output=True, text=True, check=True)
        all_interfaces = re.findall(r'^([a-zA-Z0-9]+)\s+IEEE', result.stdout, re.MULTILINE)

        if not mode_filter:
            return all_interfaces

        filtered_interfaces = []
        for iface in all_interfaces:
            try:
                iface_result = subprocess.run(['iwconfig', iface], capture_output=True, text=True, check=True)
                mode_match = re.search(r'Mode:([A-Za-z]+)', iface_result.stdout)
                if mode_match and mode_match.group(1) == mode_filter:
                    filtered_interfaces.append(iface)
            except subprocess.CalledProcessError:
                continue
        return filtered_interfaces

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error al obtener la lista de interfaces Wi-Fi: {e}[/bold red]")
        if "Operation not permitted" in str(e):
            console.print("[bold yellow]Parece que no tienes permisos. Intenta ejecutar el script con 'sudo'.[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error inesperado al listar interfaces Wi-Fi: {e}[/bold red]")
        return []

def start_tshark_capture(capture_interface: str, monitored_interface: str, name: Optional[str]) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Inicia una captura de tshark en segundo plano."""
    base_dir = "capturas"
    iface_dir = os.path.join(base_dir, get_interface_display_name(capture_interface))
    os.makedirs(iface_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    suffix = f"{name}" if name else ""
    pcap_filename = os.path.join(iface_dir, f"captura_{capture_interface}_{monitored_interface}_{suffix}_{timestamp}.pcap")

    display = get_interface_display_name(capture_interface)
    console.print(
        f"[bold green]Iniciando captura ([cyan]TShark[/cyan]) "
        f"en '[info]{display}[/] ([dim]{capture_interface}[/])'...[/bold green]"
    )
    console.print(f"[green]Archivo de captura: [blue]{os.path.abspath(pcap_filename)}[/blue][/green]")

    try:
        process = subprocess.Popen(
            ['tshark', '-i', capture_interface, '-w', pcap_filename, '-l', '-F', 'pcap'],
            preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        console.print(f"[bold green]Proceso TShark iniciado (PID: {process.pid}).[/bold green]")
        return process, pcap_filename
    except Exception as e:
        console.print(f"[bold red]Error al iniciar TShark: {e}[/bold red]")
        console.print("[bold yellow]Asegúrate de tener permisos. Intenta con 'sudo'.[/bold yellow]")
        return None, None

def stop_tshark_capture(process: Optional[subprocess.Popen]) -> None:
    """Detiene un proceso de tshark en ejecución."""
    if process and process.poll() is None:
        console.print("\n[yellow]Deteniendo captura de TShark...[/yellow]")
        try:
            os.killpg(os.getpgid(process.pid), subprocess.signal.SIGTERM)
            process.wait(timeout=5)
            console.print("[green]Captura de TShark detenida.[/green]")
        except Exception as e:
            console.print(f"[bold red]Error al detener TShark: {e}. Forzando cierre...[/bold red]")
            os.killpg(os.getpgid(process.pid), subprocess.signal.SIGKILL)