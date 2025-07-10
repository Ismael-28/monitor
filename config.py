# config.py
"""
Módulo de configuración.

Almacena constantes y configuraciones globales para la aplicación,
como el mapa de puntos de acceso conocidos.
"""

from typing import Dict
from models import PlotConfig


AP_MAP = {
    "30:DE:4B:D2:69:7B": {"name": "Nodo 1", "color": "cyan"},
    "30:DE:4B:D2:61:47": {"name": "Nodo 2", "color": "lime"},
    "30:DE:4B:D2:63:67": {"name": "Nodo 3", "color": "fuchsia"},
}

INTERFACE_MAP: Dict[str, str] = {
    "wlp0s20f3": "Intel",
    "wlx00c0cab2bc1a": "Alfa_1",
    "wlx00c0cab2bc2c": "Alfa_2",
    "wlx00c0cab3c2de": "Alfa_3",
}

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