# models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Sample:
    timestamp: datetime       # instante de la muestra
    elapsed: float            # segundos desde el inicio
    rssi: Optional[float]     # dBm
    ap_mac: str               # BSSID
    ap_name: str              # nombre legible
    latency: Optional[float]  # ms
    jitter: Optional[float]   # ms
    loss: Optional[float]     # %


@dataclass
class PlotConfig:
    ax_key: str
    title: str
    ylabel: str
    color: str
    legend_edge_color: str
    current_title: Optional[str] = None


@dataclass
class APChange:
    time: float
    name: str
