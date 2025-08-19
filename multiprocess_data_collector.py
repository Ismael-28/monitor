# -*- coding: utf-8 -*-
"""
Módulo de monitorización refactorizado con multiprocessing.

Este módulo contiene clases dedicadas para recolectar métricas de red
(RSSI, latencia, iperf3) de forma concurrente usando procesos separados.
Una clase orquestadora 'DataCollector' agrega estas métricas en una
única cola de muestras.
"""

import itertools
import multiprocessing
import re
import signal
import subprocess
import time
import netifaces
import binascii
from pyroute2 import IPRoute, IW
from pyroute2.netlink.exceptions import NetlinkError
from abc import ABC, abstractmethod
from datetime import datetime
from queue import Empty  # Usado por multiprocessing.Queue
from rich.table import Table
from typing import Any, Dict, List, Optional, Tuple

# Asumimos que estos módulos existen en tu proyecto
from models import StatusUpdate, Sample
from theme import console
from config import APS, PLOT_CONFIG
from utils import format_stat, write_log_line

class APEventCollector(multiprocessing.Process):
    """
    Se suscribe a `iw event -t` para una interfaz específica y reporta
    eventos de escaneo, conexión y desconexión con doble timestamp.
    """
    def __init__(self, interface: str):
        super().__init__(daemon=True)
        self.interface = interface
        self._stop_event = False
        self.start_time = None # Para registrar el tiempo de inicio

    def run(self):
        # Registramos el momento exacto en que el proceso comienza a ejecutarse
        self.start_time = time.monotonic()
        
        cmd = ['iw', 'event', '-t']
        try:
            # preexec_fn es para Linux/macOS. En Windows no se usa.
            # Permite matar el proceso 'iw' de forma limpia.
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, preexec_fn=subprocess.os.setsid
            )
        except (FileNotFoundError, AttributeError):
            console.print(f"[bold red]Error: El comando 'iw' no se encontró o el sistema no es compatible.[/bold red]")
            return

        for raw_line in proc.stdout:
            if self._stop_event:
                break

            line = raw_line.strip()

            # 1. Filtramos las líneas que no pertenecen a nuestra interfaz
            if not self.interface in line:
                continue

            # 2. Extraemos el timestamp del comando 'iw'
            try:
                t_iw_str, rest = line.split(':', 1)
                # El timestamp de 'iw' a veces viene con el nombre de la interfaz, lo limpiamos
                t_iw_str = t_iw_str.split()[-1]
                ts_iw = datetime.fromtimestamp(float(t_iw_str))
            except (ValueError, IndexError):
                continue # Si la línea no tiene el formato esperado, la ignoramos

            # Calculamos el tiempo transcurrido desde que se inició el programa
            elapsed_seconds = time.monotonic() - self.start_time
            
            rest = rest.strip()
            message = ""
            style = "white"

            # 3. Identificamos los eventos de interés
            if "scan started" in rest:
                message = "SCAN INICIADO"
                style = "cyan"
            elif "scan finished" in rest:
                message = "SCAN FINALIZADO"
                style = "cyan"
            elif "disconnected" in rest:
                try:
                    # Extrae BSSID tras 'disconnected'
                    message = f"ESTACIÓN DESCONECTADA"
                    style = "bold red"
                except IndexError:
                    continue
            elif "connected" in rest:
                try:
                    # Formato: connect XX:XX:XX:XX:XX:XX auth_type ...
                    bssid = rest.split("connected")[-1].strip().split()[1]
                    message = f"ESTACIÓN CONECTADA → {bssid}"
                    style = "bold green"
                except IndexError:
                    continue
            elif "del station" in rest:
                try:
                     # Formato: del station XX:XX:XX:XX:XX:XX
                    bssid = rest.split("del station")[-1].strip().split()[0]
                    message = f"ESTACIÓN BORRADA   → {bssid}"
                    style = "yellow"
                except IndexError:
                    continue

            # Si hemos identificado un evento, lo mostramos
            tiemstamp = ts_iw.strftime("%H:%M:%S.%f")[:-3]  # Formato HH:MM:SS.sss
            if message:
                console.print(
                    f"[{style}]"
                    f"{tiemstamp} " # Timestamp de 'iw'
                    f"{message}"
                    f"[/]"
                )

        # Limpieza al terminar
        proc.stdout.close()
        try:
            # Matamos el grupo de procesos para asegurar que 'iw' termine
            subprocess.os.killpg(subprocess.os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            pass

    def stop(self):
        """Marca la señal de parada y espera a que el proceso hijo termine."""
        self._stop_event = True
        console.print("\n[bold]Deteniendo el colector de eventos...[/bold]")
        self.join(timeout=2) # Damos 2 segundos para que termine limpiamente

class BaseCollector(ABC, multiprocessing.Process):
    """
    Clase base: mantiene cola, evento de parada y constructor común.
    No implementa run(), se deja a las subclases.
    """
    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.queue = multiprocessing.Queue()
        self.interval = interval
        self._stop_event = multiprocessing.Event()
        self.proc: Optional[subprocess.Popen] = None

    @abstractmethod
    def run(self) -> None:
        """
        Método que las subclases deben implementar para recolectar la métrica.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Detiene el subproceso y el proceso principal."""
        if self.proc and self.proc.poll() is None:
            console.print(f"[warn]Deteniendo subproceso de {self.__class__.__name__}...[/warn]")
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
            # 1) bloqueante: espera hasta timeout por el primer dato
            latest_item = self.queue.get_nowait()
        except Empty:
            return None

        # 2) drena todo lo que quede, quedándote con el último
        while True:
            try:
                latest_item = self.queue.get_nowait()
            except Empty:
                break

        return latest_item
    
    def flush_queue(self) -> None:
        """
        Vacía la cola tirando de todos los elementos pendientes sin bloquear.
        """
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass


class RSSICollector(BaseCollector):
    """
    Colector continuo de RSSI y MAC usando un bucle de shell.
    Cada bloque de salida de 'iwconfig' se parsea a medida que llega.
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
    def __init__(self, interface: str, target_ip: str, interval: float = 1.0):
        super().__init__(interval)
        self.interface = interface
        self.target_ip = target_ip

    def run(self) -> None:
        interface_name = 'lo' if self.target_ip == '127.0.0.1' else self.interface
        cmd = [
            'ping', '-I', interface_name, '-i', str(self.interval),
            '-s', '1400', self.target_ip
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
            m = re.search(r"time=([\d.]+)\s*ms", line)
            if m:
                self.queue.put(float(m.group(1)))

        self.proc.stdout.close()
        console.print(f"[warn]Proceso {self.__class__.__name__} finalizado.[/warn]")


class Iperf3Collector(BaseCollector):
    """
    Ejecuta un cliente iperf y recolecta jitter y pérdida de paquetes.
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

    def _parse_line(self, line: str) -> Optional[Tuple[float, float]]:
        """Parse a single iperf UDP statistics line."""
        if '0.00 bits/sec' in line:
            return None
        m = re.search(r"([\d\.]+)\s+ms\s+\d+/\d+\s+\(([0-9.eE+-]+)%\)", line)
        if not m:
            return None
        jitter = float(m.group(1))
        loss = float(m.group(2))
        return jitter, loss

    def run(self) -> None:
        """Gestiona el proceso iperf y lee su salida."""
        def get_interface_ip(iface: str) -> str:
            try:
                addrs = netifaces.ifaddresses(iface)
                return addrs[netifaces.AF_INET][0]['addr']
            except (KeyError, IndexError):
                console.print(f"[error]No se pudo obtener la IP para la interfaz {iface}.[/error]")
                return '127.0.0.1'

        src_ip = get_interface_ip(self.interface)
        self.target_ip = src_ip if self.target_ip == '127.0.0.1' else self.target_ip
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
            'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT,
            'text': True, 'bufsize': 1
        }
        if hasattr(subprocess.os, 'setsid'):
            popen_kwargs['preexec_fn'] = subprocess.os.setsid

        try:
            self.proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            console.print("[error]Comando 'iperf' no encontrado.[/error]")
            self._stop_event.set()
            return

        for line in iter(self.proc.stdout.readline, ''):
            if self._stop_event.is_set():
                break
            stats = self._parse_line(line.strip())
            if stats:
                self.queue.put(stats)
        
        self.proc.stdout.close()
        console.print(f"[warn]Proceso lector de iperf finalizado.[/warn]")


class DataCollector(BaseCollector):
    """
    Orquesta múltiples colectores para producir muestras unificadas.
    """
    def __init__(
        self,
        interface: str,
        target_ip: Optional[str] = None,
        interval: float = 1.0,
        log_file: Optional[str] = None
    ):
        super().__init__(interval)
        mgr = multiprocessing.Manager()
        self.start_time = datetime.now()
        self.sample_queue: multiprocessing.Queue[Sample] = self.queue
        self.summary_queue = multiprocessing.Queue() # Para devolver el resultado final
        self.ap_changes: list[StatusUpdate] = mgr.list()
        self._current_ap_mac: Optional[str] = None
        self.all_samples: List[Sample] = []
        self.interface = interface
        self.log_file = log_file

        self.rssi = RSSICollector(interface, interval)
        self.lat = LatencyCollector(interface, target_ip, interval) if target_ip else None
        self.ipf = Iperf3Collector(interface, target_ip, interval=interval) if target_ip else None
        # Colector de resultados de escaneo Wi-Fi cada 15 s
        self.ap_event = APEventCollector(interface)

        self.collectors = [c for c in [self.rssi, self.lat, self.ipf, self.ap_event] if c is not None]

    def _log_and_print(self, sample: Sample) -> None:
        """Imprime cada muestra formateada y la escribe en el log."""
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
        """Ejecuta el ciclo de agregación de muestras."""
        time.sleep(self.interval * 1.5)
        self.flush_all_queues()  # Vaciamos las colas al inicio para evitar datos antiguos
        while not self._stop_event.is_set():
            sample = self.collect_metric()
            if sample:
                self._log_and_print(sample)
                self.all_samples.append(sample)
                self.sample_queue.put(sample)
            time.sleep(self.interval)
        
        console.print("[warn]Proceso de recolección detenido. Enviando resumen...[/warn]")
        self.summary_queue.put(self.all_samples) # Enviar datos al proceso padre

    def start(self) -> None:
        """Inicia todos los procesos de recolección."""
        console.print(f"\nMonitorización Wi-Fi iniciada en [info]'{self.rssi.interface}'[/info]")
        console.print(f"Target: [info]{self.lat.target_ip if self.lat else 'N/A'}[/info]\n")

        for collector in self.collectors:
            collector.start()
        super().start() # Inicia el proceso de agregación de DataCollector

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
        """Detiene todos los procesos de recolección."""
        for collector in self.collectors:
            collector.stop()
        super().stop() # Detiene el proceso de agregación (setea evento y hace join)
        time.sleep(0.5)
        self.print_summary()

    def _check_ap_change(self, ap_mac: str, elapsed: float) -> None:
        """Registra cambios de AP."""
        if self._current_ap_mac is None:
            self._current_ap_mac = ap_mac
            return
        if ap_mac != self._current_ap_mac:
            change = StatusUpdate(time=elapsed, name=ap_mac)
            self.ap_changes.append(change)
            console.print(f"[warn]Cambio de AP -> {ap_mac}[/warn]")
            self._current_ap_mac = ap_mac

    def collect_metric(self) -> Optional[Sample]:
        """Agrega las últimas métricas de cada colector en un único Sample."""
        rssi_tuple = self.rssi.get_latest()
        if rssi_tuple:
            rssi, mac = rssi_tuple
            got_new_rssi = True
        else:
            rssi, mac, got_new_rssi = None, self._current_ap_mac or "", False

        latency = self.lat.get_latest() if self.lat else None
        ipf_data = self.ipf.get_latest() if self.ipf else None
        jitter, loss = ipf_data if ipf_data is not None else (None, None)

        now = datetime.now()
        elapsed = (now - self.start_time).total_seconds()

        if got_new_rssi:
            ap_mac_str = mac if mac else "Desconectado"
            self._check_ap_change(ap_mac_str, elapsed)

        ap_name = APS.get(mac, {}).get('name', mac) or 'Desconectado'

        return Sample(
            timestamp=now, elapsed=elapsed, rssi=rssi, ap_mac=mac,
            ap_name=ap_name, latency=latency, jitter=jitter, loss=loss
        )
    
    def print_summary(self) -> None:
        """Muestra un resumen con las medias de las métricas en una tabla."""
        try:
            all_samples = self.summary_queue.get(timeout=5)
        except Empty:
            console.print("[error]No se recibieron datos para el resumen.[/error]")
            all_samples = []

        if not all_samples:
            console.print("[invalid]No hay datos para calcular medias.[/]")
            return

        metrics: Dict[str, Optional[float]] = {}
        for key in PLOT_CONFIG:
            vals = [getattr(s, key) for s in all_samples if getattr(s, key) is not None]
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

        if self.ap_changes:
            changes_tbl = Table(title="Cambios de AP durante la sesión")
            changes_tbl.add_column("Tiempo (s)", style="bold", justify="right")
            changes_tbl.add_column("Nuevo AP", style="bold")
            for change in self.ap_changes:
                changes_tbl.add_row(f"{change.time:.3f}", change.name)
            console.print(changes_tbl)
        else:
            console.print("[info]No hubo cambios de AP durante la sesión.[/info]")
