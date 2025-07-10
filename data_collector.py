# -*- coding: utf-8 -*-
"""
Módulo de monitorización refactorizado.

Este módulo contiene clases dedicadas para recolectar métricas de red
(RSSI, latencia, iperf3) de forma concurrente. Una clase orquestadora
'DataCollector' agrega estas métricas en una única cola de muestras.
"""

import queue
import re
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from rich.table import Table
from typing import Any, Dict, List, Optional, Tuple

# Asumimos que estos módulos existen en tu proyecto
from models import APChange, Sample
from theme import console
from config import AP_MAP, PLOT_CONFIG
from utils import format_stat, write_log_line


class BaseCollector(ABC, threading.Thread):
    """
    Clase base: mantiene cola, evento de parada y constructor común.
    No implementa run(), se deja a las subclases.
    """
    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.interval = interval
        self._stop_event = threading.Event()
        self.proc: Optional[subprocess.Popen] = None

    @abstractmethod
    def run(self) -> None:
        """
        Método que las subclases deben implementar para recolectar la métrica.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Detiene el proceso y el hilo."""
        if self.proc and self.proc.poll() is None:
            console.print(f"[warn]Deteniendo {self.__class__.__name__}...[/warn]")
            # Enviar SIGINT al grupo de procesos
            if hasattr(subprocess.os, 'killpg'):
                subprocess.os.killpg(subprocess.os.getpgid(self.proc.pid), signal.SIGINT)
            else:
                self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                console.print(f"[error]{self.__class__.__name__} no respondió. Forzando kill.[/error]")
                self.proc.kill()
        self._stop_event.set()
        super().join(timeout=self.interval * 2)
    
    def get_latest(self) -> Optional[Any]:
        """
        Espera hasta recibir al menos un elemento (opcional timeout),
        y luego drena la cola para devolver el más reciente.
        """
        try:
            # 1) bloqueante: espera hasta timeout (o indefinido) por el primer dato
            latest_item = self.queue.get(timeout=self.interval*0.2)
        except queue.Empty:
            return None

        # 2) drena todo lo que quede, quedándote con el último
        while True:
            try:
                latest_item = self.queue.get_nowait()
            except queue.Empty:
                break

        return latest_item


class RSSICollector(BaseCollector):
    """
    Colector continuo de RSSI y MAC usando un bucle de shell.
    Cada bloque de salida de 'iwconfig' se parsea a medida que llega.
    """

    def __init__(self, interface: str, interval: float = 1.0):
        super().__init__(interval)
        self.interface = interface

    def run(self) -> None:
        # Montamos un pequeño loop en bash que emite iwconfig periódicamente
        cmd = [
            'bash', '-lc',
            f"while true; do iwconfig {self.interface}; sleep {self.interval}; done"
        ]
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
            'text': True,
            'bufsize': 1,
        }
        if hasattr(subprocess.os, 'setsid'):
            popen_kwargs['preexec_fn'] = subprocess.os.setsid

        try:
            self.proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            console.print("[error]bash o iwconfig no encontrado.[/error]")
            self._stop_event.set()
            return

        buffer = []
        for line in iter(self.proc.stdout.readline, ''):
            if self._stop_event.is_set():
                break
            # Agrupamos hasta bloque en blanco
            if line.strip() == "":
                output = "".join(buffer)
                buffer.clear()
                # Intentamos extraer RSSI y MAC; si falla, marcamos desconexión
                rssi_m = re.search(r"Signal level=(-?\d+)\s+dBm", output)
                ap_m   = re.search(r"Access Point:\s+([0-9A-Fa-f:]{17})", output)
                if rssi_m and ap_m:
                    rssi   = int(rssi_m.group(1))
                    ap_mac = ap_m.group(1).upper()
                else:
                    rssi   = None
                    ap_mac = None
                #console.print(f"Procesando salida de iwconfig: RSSI={rssi}, AP={ap_mac}")
                # Siempre emitimos un evento, aunque ap_mac sea None
                self.queue.put((rssi, ap_mac))
            else:
                buffer.append(line)
        self.proc.stdout.close()
        console.print(f"[warn]{self.__class__.__name__} finalizado.[/warn]")


class LatencyCollector(BaseCollector):
    """
    Colector continuo de latencia usando 'ping -i'.
    """

    def __init__(self, interface: str, target_ip: str, interval: float = 1.0):
        super().__init__(interval)
        self.interface = interface
        self.target_ip = target_ip

    def run(self) -> None:
        interface_name = 'lo' if self.target_ip == '127.0.0.1' else self.interface
        cmd = [
            'ping',
            '-I', interface_name,
            '-i', str(self.interval),
            '-s', '1400',
            self.target_ip
        ]
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
            'text': True,
            'bufsize': 1,
        }
        if hasattr(subprocess.os, 'setsid'):
            popen_kwargs['preexec_fn'] = subprocess.os.setsid

        try:
            self.proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            console.print("[error]Comando 'ping' no encontrado.[/error]")
            self._stop_event.set()
            return

        for line in iter(self.proc.stdout.readline, ''):
            if self._stop_event.is_set():
                break
            # Ejemplo de línea: "64 bytes from 1.2.3.4: icmp_seq=1 ttl=64 time=12.3 ms"
            m = re.search(r"time=([\d.]+)\s*ms", line)
            if m:
                latency = float(m.group(1))
                self.queue.put(latency)

        self.proc.stdout.close()
        console.print(f"[warn]{self.__class__.__name__} finalizado.[/warn]")



class Iperf3Collector(BaseCollector):
    """
    Ejecuta un cliente iperf3 y recolecta jitter y pérdida de paquetes.
    """

    def __init__(
        self,
        interface: str,
        target_ip: str,
        port: int = 5201,
        interval: float = 1.0
    ):
        super().__init__(interval)
        self.interface = interface
        self.target_ip = target_ip
        self.port = port
        self.proc: Optional[subprocess.Popen] = None

    def _parse_line(self, line: str) -> Optional[Tuple[float, float]]:
        """Parse a single iperf3 UDP statistics line."""
        if '0.00 bits/sec' in line:
            return None

        m = re.search(r"([\d\.]+)\s+ms\s+\d+/\d+\s+\(([0-9.eE+-]+)%\)", line)
        if not m:
            return None

        jitter = float(m.group(1))
        loss = float(m.group(2))

        # luego lo muestras o lo guardas como prefieras
        return jitter, loss

    def run(self) -> None:
        """
        Sobrescribe el método run para gestionar el proceso iperf3.
        El hilo se dedica a leer la salida del subproceso.
        """
        # Si el target es localhost, fuerza interfaz 'lo'
        interface_name = 'lo' if self.target_ip == '127.0.0.1' else self.interface
        cmd = [
            'iperf3',
            '-c', self.target_ip,
            '--bind-dev', interface_name,  # <--- fuerza la interfaz de salida
            '-p', str(self.port),
            '-u', '-R', '--forceflush',
            '-b', '10M',
            '-t', '3600',
            '-i', str(self.interval)
        ]
        console.print(f"Lanzando: {' '.join(cmd)}")

        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
            'text': True,
            'bufsize': 1
        }
        # preexec_fn permite matar el proceso y sus hijos fácilmente
        if hasattr(subprocess.os, 'setsid'):
            popen_kwargs['preexec_fn'] = subprocess.os.setsid

        try:
            self.proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            console.print("[error]Comando 'iperf3' no encontrado.[/error]")
            self._stop_event.set()
            return

        for line in iter(self.proc.stdout.readline, ''):
            if self._stop_event.is_set():
                break
            stats = self._parse_line(line.strip())
            if stats:
                self.queue.put(stats)
        
        self.proc.stdout.close()
        console.print("[warn]Hilo lector de iperf3 finalizado.[/warn]")

    def stop(self) -> None:
        """Envía SIGINT para una parada limpia de iperf3."""
        if self.proc and self.proc.poll() is None:
            console.print("[warn]Deteniendo iperf3...[/warn]")
            # Enviar la señal al grupo de procesos para asegurar que iperf3 la recibe
            if hasattr(subprocess.os, 'killpg'):
                subprocess.os.killpg(subprocess.os.getpgid(self.proc.pid), signal.SIGINT)
            else:
                self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                console.print("[error]iperf3 no respondió. Forzando kill.[/error]")
                self.proc.kill()
        super().stop()


class DataCollector(BaseCollector):
    """
    Orquesta múltiples colectores de métricas para producir muestras unificadas.
    """

    def __init__(
        self,
        interface: str,
        target_ip: Optional[str] = None,
        interval: float = 1.0,
        log_file: Optional[str] = None
    ):
        super().__init__(interval)
        self.start_time = datetime.now()
        self.sample_queue: queue.Queue[Sample] = self.queue  # Renombramos por claridad
        self.ap_changes: list[APChange] = []
        self._current_ap_mac: Optional[str] = None
        self.all_samples: List[Sample] = []
        self.interface = interface
        self.log_file = log_file

        # Instanciar colectores individuales
        self.rssi = RSSICollector(interface, interval)
        self.lat = LatencyCollector(interface, target_ip, interval) if target_ip else None
        self.ipf = Iperf3Collector(interface, target_ip, interval=interval) if target_ip else None

        self.collectors = [
            c for c in [self.rssi, self.lat, self.ipf]
            if c is not None
        ]

    def _log_and_print(self, sample: Sample) -> None:
        """Imprime cada muestra formateada, igual que hacías en RealTimePlot."""
        ts_fmt = sample.timestamp.strftime("%H:%M:%S.%f")[:-3]
        elapsed = f"{sample.elapsed:.3f} s"
        delta = (sample.elapsed - self._last_elapsed) if hasattr(self, "_last_elapsed") else elapsed
        delta = f"{delta:.3f} s" if isinstance(delta, float) else delta
        self._last_elapsed = sample.elapsed

        row = (
            f"[timestamp]{ts_fmt:<14}[/timestamp]| "
            f"[ap]{sample.ap_name:<19}[/ap]"
            f"[time]{elapsed:<12}[/time]"
            f"[delta]{delta:<12}[/delta]"
            f"{format_stat(sample.rssi,   '{:.0f}', ' dBm',   'rssi',    12)}"
            f"{format_stat(sample.latency,'{:.3f}', ' ms',    'latency', 14)}"
            f"{format_stat(sample.jitter, '{:.3f}', ' ms',    'jitter',  14)}"
            f"{format_stat(sample.loss,   '{:.2f}', ' %',     'loss',    10)}"
        )
        console.print(row)

        if self.log_file:
            write_log_line(self.log_file, self.interface, sample)


    def run(self) -> None:
        """
        Método requerido por BaseCollector.
        Ejecuta el ciclo de agregación de muestras.
        """
        while not self._stop_event.is_set():
            sample = self.collect_metric()
            if sample:
                self._log_and_print(sample)
                self.all_samples.append(sample)
                self.sample_queue.put(sample)
            time.sleep(self.interval)
        else:
            console.print("[warn]Hilo de recolección detenido.[/warn]")

    def start(self) -> None:
        """Inicia todos los hilos de recolección."""
        console.print(f"\nMonitorización Wi-Fi iniciada en [info]'{self.rssi.interface}'[/info]")
        console.print(f"Target: [info]{self.lat.target_ip if self.lat else 'N/A'}[/info]\n")

        for collector in self.collectors:
            collector.start()
        super().start() # Inicia el hilo de agregación de DataCollector

        header = (
            f"[timestamp]{'Hora':<14}[/timestamp]| "
            f"[ap]{'AP':<19}[/ap]"
            f"[time]{'Tiempo':<12}[/time]"
            f"[delta]{'ΔTiempo':<12}[/delta]"
            f"[rssi]{'RSSI':<12}[/rssi]"
            f"[latency]{'Latencia':<14}[/latency]"
            f"[jitter]{'Jitter':<14}[/jitter]"
            f"[loss]{'Pérdida':<10}[/loss]"
        )
        console.print(header)
        console.print("-" * 104)
        
    def stop(self) -> None:
        """Detiene todos los hilos de recolección."""
        super().stop() # Detiene el hilo de agregación
        for collector in self.collectors:
            collector.stop()
        time.sleep(0.5)  # Espera a que se detenga el hilo de muestreo
        self.print_summary()  # Muestra el resumen final

    def _check_ap_change(self, ap_mac: str, elapsed: float) -> None:
        """
        Registra cambios de AP (incluye desconexión "Desconectado" ↔ AP real),
        pero ignora la primera conexión al inicio.
        """
        # 1) Si es la primera vez (_current_ap_mac es None), sólo inicializa
        if self._current_ap_mac is None:
            self._current_ap_mac = ap_mac
            return

        # 2) Si cambia realmente, lo registras
        if ap_mac != self._current_ap_mac:
            self.ap_changes.append(APChange(time=elapsed, name=ap_mac))
            console.print(f"[warn]Cambio de AP -> {ap_mac}[/warn]")
            self._current_ap_mac = ap_mac

    def collect_metric(self) -> Optional[Sample]:
        """
        Agrega las últimas métricas de cada colector en un único Sample.
        Evita errores cuando alguno de los collectors no tiene nuevos datos.
        """
        # 1) RSSI
        rssi_tuple = self.rssi.get_latest()
        if rssi_tuple:
            rssi, mac = rssi_tuple
            got_new_rssi = True
        else:
            rssi = None
            mac = self._current_ap_mac or ""
            got_new_rssi = False

        # 2) Latencia
        latency = self.lat.get_latest() if self.lat else None

        # 3) Iperf3 (jitter y pérdida)
        ipf_data = self.ipf.get_latest() if self.ipf else None
        if ipf_data is not None:
            jitter, loss = ipf_data
        else:
            jitter, loss = None, None

        # 4) Timestamps
        now     = datetime.now()
        elapsed = (now - self.start_time).total_seconds()

        # 5) Cambio de AP solo si tenemos un nuevo MAC válido
        if got_new_rssi:
            ap_mac_str = mac if mac else "Desconectado"
            self._check_ap_change(ap_mac_str, elapsed)

        # 6) Nombre del AP
        ap_name = AP_MAP.get(mac, {}).get('name', mac) or 'Desconectado'

        # 7) Crear y devolver la muestra
        return Sample(
            timestamp=now,
            elapsed=elapsed,
            rssi=rssi,
            ap_mac=mac,
            ap_name=ap_name,
            latency=latency,
            jitter=jitter,
            loss=loss
        )
    
    def print_summary(self) -> None:
        """Muestra un resumen con las medias de las métricas en una tabla."""
        if not self.all_samples:
            console.print("[invalid]No hay datos para calcular medias.[/]")
            return

        metrics: Dict[str, Optional[float]] = {}
        for key in PLOT_CONFIG:
            vals = [getattr(s, key) for s in self.all_samples if getattr(s, key) is not None]
            metrics[key] = (sum(vals) / len(vals)) if vals else None

        table = Table(title="Resumen de la sesión")
        table.add_column("Métrica", style="bold")
        table.add_column("Media", justify="right")
        table.add_column("Unidad")
        for key, mean in metrics.items():
            if mean is not None:
                cfg = PLOT_CONFIG[key]
                ylabel = cfg.ylabel
                name = ylabel.split(' (')[0]
                unit = ylabel[ylabel.find('(')+1:ylabel.find(')')] if '(' in ylabel else ''
                colored_mean = f"[{key}_mean]{mean:.3f}[/]"
                table.add_row(name, colored_mean, unit)

        console.print(table)
