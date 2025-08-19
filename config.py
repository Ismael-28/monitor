# config.py
"""
Módulo de configuración.

Almacena constantes y configuraciones globales para la aplicación,
como el mapa de puntos de acceso conocidos.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

class APEventType(Enum):
    SCAN_STARTED   = ("Escaneo iniciado",   "wheat")
    SCAN_ABORTED   = ("Escaneo abortado",   "aquamarine")
    SCAN_FINISHED  = ("Escaneo finalizado", "plum")
    DISCONNECTED   = ("Desconectado",       "orange")
    CONNECTED      = ("Conectado",          "red")

    def __init__(self, message: str, color: str):
        self.message = message
        self.color = color

    def format(self, info: Optional[str] = None) -> str:
        return f"{self.message} -> {info}" if info else self.message


APS = {
    "30:DE:4B:D2:69:7B": {"name": "Nodo 1", "color": "cyan"},
    "30:DE:4B:D2:61:47": {"name": "Nodo 2", "color": "lime"},
    "30:DE:4B:D2:63:67": {"name": "Nodo 3", "color": "fuchsia"},
}

INTERFACES: Dict[str, str] = {
    "wlp0s20f3": "Intel",
    "wlp0s20f3mon": "Intel",
    "wlx00c0cab2bc1a": "Alfa_1",
    "wlx00c0cab2bc1amon": "Alfa_1",
    "wlx00c0cab2bc2c": "Alfa_2",
    "wlx00c0cab2bc2cmon": "Alfa_2",
    "wlx00c0cab3c2de": "Alfa_3",
    "wlx00c0cab3c2demon": "Alfa_3",
}

@dataclass
class PlotConfig:
    ax_key: str
    title: str
    ylabel: str
    color: str
    legend_edge_color: str
    current_title: Optional[str] = None

PLOT_CONFIG: Dict[str, PlotConfig] = {
    'rssi': PlotConfig(
        ax_key='ax_rssi',
        title="Intensidad de Señal",
        ylabel="RSSI (dBm)",
        color="gold",
        legend_edge_color='gold'
    ),
    'latency': PlotConfig(
        ax_key='ax_lat',
        title="Latencia",
        ylabel="Latencia (ms)",
        color="deepskyblue",
        legend_edge_color='deepskyblue'
    ),
    'jitter': PlotConfig(
        ax_key='ax_jit',
        title="Jitter",
        ylabel="Jitter (ms)",
        color="deeppink",
        legend_edge_color='deeppink'
    ),
    'loss': PlotConfig(
        ax_key='ax_loss',
        title="Tasa de Pérdidas",
        ylabel="Pérdidas (%)",
        color="blueviolet",
        legend_edge_color='blueviolet'
    )
}