# utils.py
from typing import Optional
from models import Sample
from config import INTERFACE_MAP

def get_interface_display_name(iface: str) -> str:
    """Devuelve nombre amigable si existe, o el nombre real."""
    return INTERFACE_MAP.get(iface, iface)


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

def build_monitor_output(sample: Sample) -> str:
    """
    Construye una línea de estado a partir de un Sample, con todas las métricas alineadas.
    """
    # ancho fijo para cada métrica
    W = 10

    ts = sample.timestamp.strftime("%H:%M:%S.%f")[:-3]
    rssi_str = format_stat(sample.rssi, "{:.0f}", " dBm", "rssi", W)
    lat_str  = format_stat(sample.latency, "{:.2f}", " ms", "latency", W)
    jit_str  = format_stat(sample.jitter, "{:.2f}", " ms", "jitter", W)
    loss_str = format_stat(sample.loss, "{:.2f}", "%", "loss", W)

    base = (
        f"[cyan][{ts}][/cyan] "
        f"AP: [magenta]{sample.ap_name:<15}[/magenta] "
        f"RSSI: {rssi_str} "
        f"Latencia: {lat_str} "
        f"Jitter: {jit_str} "
        f"Pérdida: {loss_str}"
    )

    return base

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
