# monitoring.py
"""
Módulo de monitorización.

Contiene la lógica para obtener información de la red Wi-Fi y para
ejecutar el modo de monitorización en texto plano.
"""

import subprocess
import re
import signal
import threading
import time
import queue
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from models import APChange, Sample
from theme import console

from config import AP_MAP

def get_ping_latency(target_ip: str, count: int = 1) -> Optional[float]:
    try:
        cmd = ['ping', '-c', str(count), '-s', '1400', '-W', '1', target_ip]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        output = result.stdout
        m = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", output)
        if m:
            return float(m.group(1))
        if "Destination Host Unreachable" in output:
            console.print(f"[warn]Host {target_ip} inalcanzable.[/warn]")
        elif "100% packet loss" in output:
            console.print(f"[warn]100% de pérdida de paquetes a {target_ip}.[/warn]")
        return None
    except subprocess.TimeoutExpired:
        console.print(f"[warn]El ping a {target_ip} excedió el tiempo de espera.[/warn]")
        return None
    except Exception as e:
        console.print(f"[error]Error al ejecutar ping a {target_ip}: {e}[/error]")
        return None


def get_wifi_info(interface: str, target: Optional[str] = None) -> Dict[str, Any]:
    rssi = None
    ap_mac = "No conectado"
    latency = None

    try:
        result = subprocess.run(['iwconfig', interface], capture_output=True, text=True, check=True)
        output = result.stdout
        m_rssi = re.search(r"Signal level=(-?\d+)\s+dBm", output)
        if m_rssi:
            rssi = int(m_rssi.group(1))
        m_ap = re.search(r"Access Point:\s+([0-9A-Fa-f:]{17})", output)
        if m_ap:
            ap_mac = m_ap.group(1).upper()
    except Exception:
        pass

    if target:
        latency = get_ping_latency(target)

    return {"rssi": rssi, "ap_mac": ap_mac, "latency": latency}


def parse_iperf3_udp_stats_line(line: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse jitter y pérdida de una línea de iperf3 UDP.

    Args:
        line: Una línea de salida de iperf3 UDP.

    Returns:
        Un par (jitter_ms, loss_pct), o (None, None) si no coincide.
    """
    m = re.search(r"([\d\.]+)\s+ms\s+\d+/\d+\s+\(([0-9\.]+)%\)", line)
    console.print(f"[debug]Parsing iperf3 line: {line}")
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


class Iperf3Client:
    """
    Cliente iperf3 UDP reversible (-R) que corre en background,
    mete cada stat (jitter, pérdida) en una cola y permite leerla
    sin bloquear.
    """
    def __init__(self,
        server_ip: str,
        port: int = 5201,
        interval: float = 1.0,
        duration: float = 1800.0
    ) -> None:
        self.server_ip = server_ip
        self.port = port
        self.interval = interval
        self.duration = duration
        self.proc = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._stopper_thread = threading.Thread(target=self._stopper, daemon=True)

    def start(self) -> Optional["Iperf3Client"]:
        cmd = [
            'iperf3', '-c', self.server_ip, '-p', str(self.port),
            '-u', '-R', '--forceflush', '-b', '10M',  # ancho de banda de 10 Mbps
            '-t', str(self.duration), '-i', str(self.interval)
        ]
        console.print(f"Lanzando iperf3 a {self.server_ip}:{self.port} "
                      f"(interval={self.interval}s, duration={self.duration}s)")

        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
            'text': True,
            'bufsize': 1
        }
        if subprocess.os.name != 'nt':
            popen_kwargs['preexec_fn'] = subprocess.os.setsid

        try:
            self.proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            console.print("[error]Error: no se encontró 'iperf3'.[/error]")
            return None
        except Exception as e:
            console.print(f"[error]Error al lanzar iperf3: {e}[/error]")
            return None

        # Iniciamos hilos
        self._reader_thread.start()
        self._stopper_thread.start()

        return self

    def stop(self) -> None:
        """Envía SIGINT y, si es necesario, SIGKILL."""
        if not self.proc or self.proc.poll() is not None:
            return
        self._stop_event.set()
        console.print("[warn]Deteniendo iperf3...[/warn]")
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=5)
            console.print("[success]iperf3 detenido correctamente.[/success]")
        except subprocess.TimeoutExpired:
            console.print("[error]No respondió a SIGINT; forzando kill.[/error]")
            self.proc.kill()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[float, float]:
        """
        Extrae el siguiente par (jitter_ms, loss_pct) de la cola.
        Si block=True, espera hasta timeout (o indefinido).
        """
        return self._queue.get(block, timeout)

    def empty(self) -> bool:
        """True si no hay stats pendientes."""
        return self._queue.empty()

    def is_running(self) -> bool:
        """True mientras el proceso siga vivo."""
        return self.proc and self.proc.poll() is None
    
    def _reader(self) -> None:
        """Lee stdout, parsea y mete stats en la cola."""
        for raw in self.proc.stdout:
            if self._stop_event.is_set():
                break
            line = raw.strip()
            jitter, loss = parse_iperf3_udp_stats_line(line)
            if jitter is not None:
                self._queue.put((jitter, loss))
        console.print("[warn]Hilo de lectura finalizado.[/warn]")

    def _stopper(self) -> None:
        """Detiene el proceso tras el tiempo indicado."""
        time.sleep(self.duration)
        self.stop()


class DataCollector:
    """
    Recoge muestras de Wi-Fi y estadísticas de iperf3 de forma desacoplada.
    Tiene un hilo propio que genera objetos Sample y los deposita en una cola.
    """
    def __init__(
        self,
        interface: str,
        target: Optional[str] = None,
        interval: float = 1.0,
        duration: float = 3600.0
    ) -> None:
        self.interface = interface
        self.target = target
        self.interval = interval
        self.duration = duration

        # Cola para pasar muestras al consumidor
        self.queue: queue.Queue[Sample] = queue.Queue()
        # Evento de parada
        self._stop_event = threading.Event()
        # Hiló de recolección
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Cliente Iperf3:
        self.iperf_client = Iperf3Client(
            server_ip=target or '',
            port=5201,
            interval=interval,
            duration=duration
        ) if target else None

        # Estado para detección de cambios de AP
        self._current_ap: Optional[str] = None
        self.ap_changes: list[APChange] = []

    def start(self) -> None:
        """Inicia el hilo de recolección (y iperf3)."""
        if self.iperf_client:
            self.iperf_client.start()
            # Pequeña espera para primer informe
            time.sleep(self.interval)
        self._thread.start()

    def stop(self) -> None:
        """Señaliza la parada y espera a que termine el hilo."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        if self.iperf_client:
            self.iperf_client.stop()

    def _run(self) -> None:
        """Bucle principal: recoge muestras y las mete en la cola."""
        start_time = datetime.now()
        while not self._stop_event.is_set():
            now = datetime.now()
            elapsed = (now - start_time).total_seconds()

            # Datos de Wi-Fi
            info = get_wifi_info(self.interface, self.target)
            rssi = info.get('rssi')
            ap_mac = info.get('ap_mac', 'Desconocido')
            latency = info.get('latency')
            ap_name = AP_MAP.get(ap_mac, {}).get('name', ap_mac)

            # Estadísticas iperf
            jitter = loss = None
            if self.iperf_client and not self.iperf_client.empty():
                try:
                    jitter, loss = self.iperf_client.get(block=False)
                except queue.Empty:
                    pass

            # Detectar cambio de AP
            if self._current_ap and ap_mac != self._current_ap:
                change = APChange(time=elapsed, name=ap_name)
                self.ap_changes.append(change)
            self._current_ap = ap_mac

            # Crear muestra y encolar
            sample = Sample(
                timestamp=now,
                elapsed=elapsed,
                rssi=rssi,
                ap_mac=ap_mac,
                ap_name=ap_name,
                latency=latency,
                jitter=jitter,
                loss=loss
            )
            self.queue.put(sample)

            # Espera siguiente iteración
            time.sleep(self.interval)

    def get_sample(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Sample]:
        """Extrae la siguiente Sample de la cola. Devuelve None si no hay datos."""
        try:
            return self.queue.get(block, timeout)
        except queue.Empty:
            return None
