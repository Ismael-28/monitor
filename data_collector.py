# -*- coding: utf-8 -*-
"""
Módulo de monitorización refactorizado.

Este módulo contiene clases dedicadas para recolectar métricas de red
(RSSI, latencia, iperf3) de forma concurrente. Una clase orquestadora
'DataCollector' agrega estas métricas en una única cola de muestras.
"""

from concurrent.futures import ThreadPoolExecutor
import queue
import re
import signal
import subprocess
import threading
import time
from pyroute2 import IPRoute, IW
from pyroute2.netlink.exceptions import NetlinkError
from abc import ABC, abstractmethod
from datetime import datetime
from rich.table import Table
from typing import Any, Dict, List, Optional, Tuple
from models import StatusUpdate, Sample, Event
from theme import console
from config import APS, PLOT_CONFIG, APEventType
from utils import format_stat, get_ap_display_name, write_log_line

class APEventCollector(threading.Thread):
    """
    Se suscribe a `iw event -t` y publica en event_queue:
      - ESCANEO INICIADO / ESCANEO FINALIZADO
      - ESTACIÓN DESCONECTADA / ESTACIÓN CONECTADA -> BSSID
      - ESTACIÓN BORRADA   -> BSSID
      - DURACIÓN ESCANEO: XX ms
      - TIEMPO RECONEXIÓN: XX ms
    """
    def __init__(
            self,
            interface: str,
            event_queue: queue.Queue[Event],
            status_updates: list[StatusUpdate],
            start_time: datetime
        ):
        super().__init__(daemon=True)
        self.interface = interface
        self.event_queue = event_queue
        self.status_updates   = status_updates
        self.start_time   = start_time
        self._stop_event = False

        # Tiempos para cálculo de duraciones
        self._scan_start_ts = None
        self._del_station_ts = None

    def run(self):
        cmd = ['iw', 'event', '-t']
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                preexec_fn=subprocess.os.setsid
            )
        except (FileNotFoundError, AttributeError):
            console.print("[error]Error: comando 'iw' no encontrado o sistema no compatible[/error]")
            return

        for raw_line in proc.stdout:
            if self._stop_event:
                break

            line = raw_line.strip()
            if self.interface not in line:
                continue

            # extraer timestamp de 'iw'
            try:
                t_iw_str, rest = line.split(':', 1)
                t_iw_str = t_iw_str.split()[-1]
                ts_iw = datetime.fromtimestamp(float(t_iw_str))
            except (ValueError, IndexError):
                continue

            rest = rest.strip().lower()
            elapsed = (ts_iw - self.start_time).total_seconds()

            if "scan started" in rest:
                self.event_queue.put(Event(ts_iw, "Escaneo iniciado"))
                self.status_updates.append(StatusUpdate(
                    time=elapsed,
                    event_type=APEventType.SCAN_STARTED
                ))
                self._scan_start_ts = ts_iw

            if "scan aborted" in rest:
                self.event_queue.put(Event(ts_iw, "Escaneo abortado"))
                self.status_updates.append(StatusUpdate(
                    time=elapsed,
                    event_type=APEventType.SCAN_ABORTED
                ))
                if self._scan_start_ts:
                    dur = (ts_iw - self._scan_start_ts).total_seconds() * 1000
                    # evento extra con duración en ms
                    self.event_queue.put(Event(ts_iw, f"Duración escaneo: {dur:.3f} ms"))
                    self._scan_start_ts = None
                    
            elif "scan finished" in rest:
                self.event_queue.put(Event(ts_iw, "Escaneo finalizado"))
                self.status_updates.append(StatusUpdate(
                    time=elapsed,
                    event_type=APEventType.SCAN_FINISHED
                ))
                if self._scan_start_ts:
                    dur = (ts_iw - self._scan_start_ts).total_seconds() * 1000
                    # evento extra con duración en ms
                    self.event_queue.put(Event(ts_iw, f"Duración escaneo: {dur:.3f} ms"))
                    self._scan_start_ts = None

            elif "disconnected" in rest:
                self.event_queue.put(Event(ts_iw, "Estación desconectada"))
                self.status_updates.append(StatusUpdate(
                    time=elapsed,
                    event_type=APEventType.DISCONNECTED
                ))

            elif "del station" in rest:
                # Formato: del station XX:XX:XX:XX:XX:XX
                parts = rest.split()
                if len(parts) >= 3:
                    bssid = parts[3].upper()
                    ap_name = get_ap_display_name(bssid)
                    self.event_queue.put(Event(ts_iw, f"Estación borrada -> {ap_name}"))
                    self._del_station_ts = ts_iw

            elif "connected" in rest:
                # Formato: connect XX:XX:XX:XX:XX:XX auth_type...
                parts = rest.split()
                if len(parts) >= 5:
                    bssid = parts[5].upper()
                    ap_name = get_ap_display_name(bssid)
                    self.event_queue.put(Event(ts_iw, f"Estación conectada -> {ap_name}"))
                    # registro conexión
                    self.status_updates.append(StatusUpdate(
                        time=elapsed,
                        event_type=APEventType.CONNECTED,
                        info=ap_name
                    ))
                    if self._del_station_ts:
                        recon = (ts_iw - self._del_station_ts).total_seconds() * 1000
                        self.event_queue.put(Event(ts_iw, f"Tiempo reconexión: {recon:.3f} ms"))
                        self._del_station_ts = None

        proc.stdout.close()
        try:
            subprocess.os.killpg(subprocess.os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            pass

    def stop(self):
        self._stop_event = True
        console.print("\n[bold]Deteniendo colector de eventos...[/bold]")
        self.join(timeout=2)

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
            latest_item = self.queue.get_nowait()
        except queue.Empty:
            return None

        # 2) drena todo lo que quede, quedándote con el último
        while True:
            try:
                latest_item = self.queue.get_nowait()
            except queue.Empty:
                break

        return latest_item
    
    def flush_queue(self) -> None:
        """
        Vacía la cola tirando de todos los elementos pendientes sin bloquear.
        """
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass


class RSSICollector(BaseCollector):
    """
    Colector continuo de RSSI y MAC usando un bucle de shell.
    Cada bloque de salida de 'iw' se parsea a medida que llega.
    """

    def __init__(self, interface: str, interval: float = 1.0):
        super().__init__(interval)
        self.interface = interface
        self.iw = IW()  # Mantenemos una instancia de IW para toda la vida del objeto
        
        # Obtenemos el índice de la interfaz una sola vez durante la inicialización
        try:
            ipr = IPRoute()
            self.interface_idx = ipr.link_lookup(ifname=interface)[0]
            ipr.close()
        except IndexError:
            console.print(f"[error]Interfaz '{self.interface}' no encontrada.[/error]")
            # Marcamos el evento de parada si la interfaz no existe para detener la ejecución
            self._stop_event.set()
        except Exception as e:
            console.print(f"[error]Error al inicializar: {e}[/error]")
            self._stop_event.set()

    def _get_wifi_stats(self) -> Tuple[Optional[int], Optional[str]]:
        """
        Obtiene RSSI y BSSID. Devuelve (None, None) si no está conectado
        o si falta cualquier parte de la estructura esperada.
        """
        try:
            bss_message = self.iw.get_associated_bss(self.interface_idx)
            # 1) Validar que existe y tiene attrs
            if not bss_message or 'attrs' not in bss_message:
                return None, None

            # 2) Convertir la lista de attrs a dict
            attrs = dict(bss_message.get('attrs', []))

            # 3) Extraer la parte BSS y validar
            bss_section = attrs.get('NL80211_ATTR_BSS')
            if not bss_section or not isinstance(bss_section, dict):
                return None, None

            # 4) Convertir sus attrs anidados a dict
            nested = dict(bss_section.get('attrs', []))

            # 5) Obtener señal (puede venir como dict o tuple)
            signal_mbm = nested.get('NL80211_BSS_SIGNAL_MBM')
            if isinstance(signal_mbm, dict):
                signal_mbm = signal_mbm.get('VALUE')
            elif isinstance(signal_mbm, tuple) and len(signal_mbm) == 2:
                _, signal_mbm = signal_mbm
            # 6) Obtener BSSID
            raw_bssid = nested.get('NL80211_BSS_BSSID')

            # 7) Si falta algo, devolvemos desconectado
            if signal_mbm is None or raw_bssid is None:
                return None, None

            # 8) Convertir mBm → dBm
            rssi = int(signal_mbm / 100.0)

            # 9) Formatear MAC según tipo
            if isinstance(raw_bssid, (bytes, bytearray)):
                ap_mac = ':'.join(f"{b:02X}" for b in raw_bssid)
            else:
                ap_mac = str(raw_bssid).upper()

            return rssi, ap_mac

        except (KeyError, NetlinkError):
            # Estructura inesperada o interfaz no asociada
            return None, None

        except Exception as e:
            console.print(f"[error]Error inesperado en _get_wifi_stats: {e}[/error]")
            return None, None

    def run(self) -> None:
        """
        Bucle principal del colector. Se ejecuta hasta que se activa el evento de parada.
        """
        # Si la inicialización falló (ej. interfaz no encontrada), salimos inmediatamente.
        if self._stop_event.is_set():
            console.print(f"[warn]Proceso {self.__class__.__name__} no iniciado debido a un error de inicialización.[/warn]")
            self.iw.close()
            return
            
        while not self._stop_event.is_set():
            # Obtenemos los datos llamando a nuestro método auxiliar
            rssi, ap_mac = self._get_wifi_stats()
            
            # Ponemos el resultado en la cola, sea válido o (None, None)
            self.queue.put((rssi, ap_mac))
            
            # Esperamos el intervalo de tiempo definido.
            # a diferencia de time.sleep(), wait() es sensible al evento
            # de parada, por lo que el colector se detendrá más rápido.
            self._stop_event.wait(self.interval)
            
        self.iw.close() # Liberamos el recurso al finalizar
        console.print(f"[warn]Proceso {self.__class__.__name__} finalizado.[/warn]")


class LatencyCollector(BaseCollector):
    """
    Colector continuo de latencia usando 'ping -i'.
    """
    LATENCY_THRESHOLD_MS = 150

    def __init__(
        self,
        interface: str,
        target_ip: str,
        event_queue: queue.Queue[Event],
        interval: float = 1.0
    ):
        super().__init__(interval)
        self.interface = interface
        self.target_ip = target_ip
        self.event_queue = event_queue

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
                if latency > self.LATENCY_THRESHOLD_MS:
                    self.event_queue.put(
                        Event(
                            datetime.now(),
                            f"Latencia alta: {latency:.3f} ms"
                        )
                    )

        self.proc.stdout.close()
        console.print(f"[warn]{self.__class__.__name__} finalizado.[/warn]")



class Iperf3Collector(BaseCollector):
    """
    Ejecuta un cliente iperf3 y recolecta jitter y pérdida de paquetes.
    """

    def __init__(
        self,
        event_queue: queue.Queue[Event],
        interface: str,
        target_ip: str,
        port: int = 5201,
        interval: float = 1.0
    ):
        super().__init__(interval)
        self.interface = interface
        self.target_ip = target_ip
        self.port = port
        self.event_queue = event_queue
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
        if loss > 30:
            self.event_queue.put(Event(datetime.now(), f"Pérdida: {loss:.2f}%"))

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
            '-t', '300',
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
        self.status_changes: list[StatusUpdate] = []
        self._current_ap_mac: Optional[str] = None
        self.all_samples: List[Sample] = []
        self.interface = interface
        self.log_file = log_file
        self.event_queue: queue.Queue[Event] = queue.Queue()
        self.event_list: List[Event] = []

        # Instanciar colectores individuales
        self.rssi = RSSICollector(interface, interval)
        self.lat = LatencyCollector(interface, target_ip, self.event_queue, interval) if target_ip else None
        self.ipf = Iperf3Collector(self.event_queue, interface, target_ip, interval=interval) if target_ip else None
        self.ap_event = APEventCollector(interface, self.event_queue, self.status_changes, self.start_time)

        self.collectors = [
            c for c in [self.rssi, self.lat, self.ipf, self.ap_event]
            if c is not None
        ]

        self.executor = ThreadPoolExecutor(max_workers=len(self.collectors))
        self._event_printer_thread = threading.Thread(target=self._event_printer, daemon=True)

    
    def _event_printer(self) -> None:
        """
        Hilo que imprime cada evento tan pronto como llegue al canal.
        """
        while not self._stop_event.is_set():
            try:
                ev = self.event_queue.get()
            except queue.Empty:
                continue
            self.event_list.append(ev)
            console.print(f"[timestamp]{ev.ts}[/]  | [warn]{ev.msg}[/]")

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

    def flush_all_queues(self) -> None:
        # La propia DataCollector hereda BaseCollector, así que vacía su queue...
        self.flush_queue()
        # …y vacía las de cada sub‐colector:
        for c in self.collectors:
            if isinstance(c, BaseCollector):
                c.flush_queue()


    def run(self) -> None:
        """
        Método requerido por BaseCollector.
        Ejecuta el ciclo de agregación de muestras.
        """
        self.flush_all_queues()  # Vaciamos las colas al inicio para evitar datos antiguos
        time.sleep(self.interval * 1.5)
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
        self._event_printer_thread.start()  # Inicia el hilo de impresión de eventos

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
        self.executor.shutdown(wait=True)
        time.sleep(0.5)  # Espera a que se detenga el hilo de muestreo
        self.print_summary()  # Muestra el resumen final

    def collect_metric(self) -> Optional[Sample]:
        """
        Agrega las últimas métricas de cada colector en un único Sample,
        lanzando las llamadas en paralelo.
        """
        # Lanzamos todas las tareas simultáneamente
        futures = {
            'rssi':    self.executor.submit(self.rssi.get_latest),
        }
        if self.lat:
            futures['latency'] = self.executor.submit(self.lat.get_latest)
        if self.ipf:
            futures['ipf']     = self.executor.submit(self.ipf.get_latest)

        # 1) RSSI
        rssi_tuple = futures['rssi'].result()
        if rssi_tuple:
            rssi, mac = rssi_tuple
            got_new_rssi = True
        else:
            rssi = None
            mac = self._current_ap_mac or ""
            got_new_rssi = False

        # 2) Latencia
        latency = futures.get('latency').result() if 'latency' in futures else None

        # 3) Iperf3 (jitter y pérdida)
        ipf_data = futures.get('ipf').result() if 'ipf' in futures else None
        if ipf_data is not None:
            jitter, loss = ipf_data
        else:
            jitter, loss = None, None

        # 4) Timestamps
        now     = datetime.now()
        elapsed = (now - self.start_time).total_seconds()

        # 5) Cambio de AP si hay nuevo MAC
        ap_mac_str = mac if mac else "Desconectado"

        if ap_mac_str != self._current_ap_mac:
            self._current_ap_mac = ap_mac_str

        # 6) Nombre del AP
        ap_name = get_ap_display_name(mac)

        # 7) Mostrar sample
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

        if self.status_changes:
            changes_tbl = Table(title="Cambios de estado durante la sesión")
            changes_tbl.add_column("Tiempo (s)", style="bold", justify="right")
            changes_tbl.add_column("Cambios de estado", style="bold")
            for change in self.status_changes:
                changes_tbl.add_row(f"{change.time:.3f}", change.name)
            console.print(changes_tbl)
        else:
            console.print("[info]No hubo cambios de AP durante la sesión.[/info]")