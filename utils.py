# utils.py

import os
from datetime import datetime
from typing import Optional
from models import Sample
from config import APS, INTERFACES

def get_interface_display_name(iface: str) -> str:
    """Devuelve nombre amigable si existe, o el nombre real."""
    return INTERFACES.get(iface, iface)

def get_ap_display_name(bssid: str) -> str:
    """
    Devuelve el nombre amigable del AP si existe, o el BSSID real.
    """
    return APS.get(bssid, {}).get('name', bssid) or 'Desconectado'

def format_stat(value: Optional[float], fmt: str, unit: str, color: str, width: int) -> str:
    """
    - value: el valor numérico, o None.
    - fmt: formato estilo '{:.2f}' antes de la unidad.
    - unit: sufijo (p.ej. ' ms', ' dBm', '%').
    - color: nombre de color Rich.
    - width: ancho fijo de caracteres del texto visible.
    """
    raw = "N/A" if value is None else fmt.format(value) + unit
    padded = raw.ljust(width)
    return f"[{color}]{padded}[/{color}]"


def write_log_line(log_file, interface: str, sample: Sample) -> None:
    """
    Escribe una línea de datos en el archivo de log CSV a partir de un Sample.
    """
    ts       = sample.timestamp.strftime("%H:%M:%S.%f")[:-3]
    rssi_str = f"{sample.rssi}"    if sample.rssi    is not None else ""
    lat_str  = f"{sample.latency:.3f}" if sample.latency is not None else ""
    jit_str  = f"{sample.jitter:.3f}"  if sample.jitter  is not None else ""
    loss_str = f"{sample.loss:.3f}"    if sample.loss    is not None else ""

    line = (
        f"{ts},{interface},{sample.ap_name},"
        f"{rssi_str},{lat_str},{jit_str},{loss_str}\n"
    )
    log_file.write(line)
    log_file.flush()


def build_filepath(
    category: str,
    interface: str,
    name: Optional[str],
    timestamp: datetime,
    prefix: Optional[str] = None,
    extension: Optional[str] = None,
) -> str:
    """
    Construye y crea (si hace falta) el path completo para un fichero.

    Args:
        category: tipo de recurso, p.ej. 'graficas', 'datos_graficas', 'logs', 'capturas'...
        interface: nombre de la interfaz (p.ej. 'wlan0').
        name: etiqueta adicional (o None para omitir).
        timestamp: objeto datetime para timestamp; si es None usa ahora().
        prefix: prefijo para el nombre de fichero (p.ej. 'grafica', 'datos', 'mon', 'captura').
        extension: extensión sin punto (p.ej. 'png', 'csv', 'log', 'pcap'); si None, no añade.
    
    Devuelve:
        Ruta absoluta al fichero (sin crearlo aún), asegurando que existe la carpeta.
    """
    # 1) Timestamp
    date = timestamp.date().strftime('%Y-%m-%d')
    time = timestamp.time().strftime('%H-%M-%S')
    interface = get_interface_display_name(interface)

    # 2) Carpeta raíz y de interfaz
    iface_dir = os.path.join(date, category, interface)

    # 3) Asegurar que existe
    os.makedirs(iface_dir, exist_ok=True)

    # 4) Montar nombre de fichero
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(interface)
    if name:
        parts.append(name)
    parts.append(time)
    filename = "_".join(parts)

    if extension:
        filename = f"{filename}.{extension.lstrip('.')}"
    
    # 5) Ruta completa
    return os.path.join(iface_dir, filename)
