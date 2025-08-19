# models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import APEventType


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
class StatusUpdate:
    time: float
    event_type: APEventType
    info: str = ""     # p.ej. el SSID o la duración en ms

    @property
    def name(self) -> str:
        # para mantener compatibilidad con el código que lee `change.name`
        return self.event_type.format(self.info)
    
    def parse_status_change_entry(entry: dict) -> 'StatusUpdate':
        """
        Convierte un dict {'time': float, 'name': str} en un StatusUpdate,
        extrayendo event_type e info a partir de name.
        """
        raw_name = entry['name']
        for evt in APEventType:
            # formatea sin info para obtener solo el mensaje base
            base = evt.format("")        # ej. "Conectado" o "Desconectado"
            # 1) Caso sin info añadida
            if raw_name == base:
                return StatusUpdate(time=entry['time'], event_type=evt, info="")
            # 2) Caso con info: busca el separador " -> "
            sep = f"{base} -> "
            if raw_name.startswith(sep):
                info = raw_name[len(sep):]  # todo lo que viene después
                return StatusUpdate(time=entry['time'], event_type=evt, info=info)
        # si no empareja ninguno, lanzamos
        raise ValueError(f"Nombre de evento desconocido: '{raw_name}'")


@dataclass
class Event:
    _ts: datetime         = field(repr=False)
    msg: str

    @property
    def ts(self) -> str:
        """Cada vez que accedes a .ts te devuelve el string con ms."""
        return self._ts.strftime("%H:%M:%S.%f")[:-3]

    @ts.setter
    def ts(self, new_ts: datetime) -> None:
        """Te permite asignar un datetime a .ts."""
        self._ts = new_ts