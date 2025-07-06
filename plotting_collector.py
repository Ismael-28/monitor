import os
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

from rich.table import Table
import pandas as pd
import json
from theme import console

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    plt.style.use('dark_background')
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    console.print(f"[yellow]Advertencia: no se pudo importar Matplotlib ({e}). Las funciones de graficación están desactivadas.[/yellow]")
    MATPLOTLIB_AVAILABLE = False

from config import AP_MAP, PLOT_CONFIG
from monitoring import get_wifi_info, Iperf3Client
from data_collector import DataCollector
from utils import build_monitor_output, format_stat, get_interface_display_name, write_log_line
from models import Sample, APChange


# -----------------------------------------------------------------------------
# Funciones de Ayuda para Trazado
# -----------------------------------------------------------------------------

def style_axis(
    ax: Axes,
    title: str,
    ylabel: str,
    color: str,
    xlabel: Optional[str] = None,
) -> None:
    """
    Aplica un estilo coherente a un Axes de Matplotlib:
      - Título en blanco con tamaño fijo
      - Etiqueta de eje y en `color`
      - Grid dashed
      - Espinas coloreadas
      - (Opcional) Etiqueta de eje x
    """
    ax.set_title(title, fontsize=14, color='white')
    ax.set_ylabel(ylabel, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='y')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)

def _setup_plot_axes(
    fig: Figure,
) -> Dict[str, Axes]:
    """Crea y estila los 4 ejes a partir de PLOT_CONFIG."""
    axes_map = {
        'ax_rssi': fig.add_subplot(2, 2, 1),
        'ax_lat':  fig.add_subplot(2, 2, 2),
        'ax_jit':  fig.add_subplot(2, 2, 3),
        'ax_loss': fig.add_subplot(2, 2, 4)
    }

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.93,
        bottom=0.07,
        hspace=0.3,
        wspace=0.15
    )

    # Asigna títulos dinámicos
    latency_cfg = PLOT_CONFIG['latency']
    jitter_cfg  = PLOT_CONFIG['jitter']
    latency_cfg.current_title = latency_cfg.title
    jitter_cfg.current_title  = jitter_cfg.title

    for config in PLOT_CONFIG.values():
        ax = axes_map[config.ax_key]
        title = config.current_title or config.title
        style_axis(ax, title, config.ylabel, config.color, xlabel="Tiempo (s)")

    axes_map['ax_loss'].set_ylim(bottom=-10, top=100)
    return axes_map


def _draw_ap_change_annotations(ax: Axes, ap_changes: List[APChange]):
    """Dibuja anotaciones de cambio de AP en el eje `ax`."""
    # Limpiar solo las anotaciones anteriores para redibujar
    for artist in list(ax.artists):
        if hasattr(artist, 'get_label') and artist.get_label() == '_ap_change_annotation':
            artist.remove()
    for line in list(ax.lines):
        if hasattr(line, 'get_label') and line.get_label() == '_ap_change_vline':
            line.remove()

    y_min, y_max = ax.get_ylim()
    text_y = y_min + (y_max - y_min) * 0.05
    for change in ap_changes:
        ax.axvline(
            x=change.time,
            color='red',
            linestyle='--',
            linewidth=1.5,
            zorder=0,
            label='_ap_change_vline'
        )
        ax.text(
            change.time + 0.5,
            text_y,
            f"->{change.name}",
            color='white',
            rotation=90,
            va='bottom',
            fontsize=9,
            bbox=dict(boxstyle='round', fc='red', alpha=0.7, ec='none'),
            zorder=10,
            label='_ap_change_annotation'
        )

# -----------------------------------------------------------------------------
# Funciones de Exportación
# -----------------------------------------------------------------------------


def generate_final_plot(
    interface: str,
    samples: List[Sample],
    ap_changes: List[APChange],
    start_time_str: str,
    figsize: Tuple[int,int],
    target: str,
) -> Optional[Figure]:
    if not MATPLOTLIB_AVAILABLE or not samples:
        return None

    fig = plt.figure(figsize=figsize)
    display = get_interface_display_name(interface)
    fig.suptitle(
        f"Monitor Wi-Fi: {display} ({interface}) (Inicio: {start_time_str})",
        fontsize=16, color='white', weight='bold'
    )
    axes = _setup_plot_axes(fig)
    ax_rssi = axes['ax_rssi']
    ax_lat  = axes['ax_lat']
    ax_jit  = axes['ax_jit']
    ax_loss = axes['ax_loss']

    # --- Trazado de RSSI por AP con cortes en discontinuidades ---
    rssi_data: Dict[str, Dict[str, List]] = {}
    last_mac: Optional[str] = None
    for s in samples:
        if s.rssi is None:
            continue
        mac = s.ap_mac
        rssi_data.setdefault(mac, {'x': [], 'y': []})
        # Insertar None al cambiar de AP para cortar la línea
        if last_mac is not None and mac != last_mac and rssi_data[mac]['x']:
            rssi_data[mac]['x'].append(None)
            rssi_data[mac]['y'].append(None)
        rssi_data[mac]['x'].append(s.elapsed)
        rssi_data[mac]['y'].append(s.rssi)
        last_mac = mac

    for mac, data in rssi_data.items():
        if not data['x']:
            continue
        cfg = AP_MAP.get(mac, {})
        label = cfg.get("name", f"AP Desc ({mac})")
        color = cfg.get("color", "gray")
        ls = '-' if mac in AP_MAP else ':'
        mk = '.' if mac in AP_MAP else 'x'
        ax_rssi.plot(
            data['x'], data['y'],
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=4,
            label=label
        )


    # --- Trazado de Latencia, Jitter, Pérdida (con cortes en cambios de AP) ---
    for metric_key, config in PLOT_CONFIG.items():
        if metric_key == 'rssi':
            continue
        ax = axes[config.ax_key]
        xs, ys = [], []
        last_mac = samples[0].ap_mac if samples else None
        for s in samples:
            if s.ap_mac != last_mac:
                xs.append(s.elapsed)
                ys.append(None)
            xs.append(s.elapsed)
            ys.append(getattr(s, metric_key))
            last_mac = s.ap_mac
        if any(v is not None for v in ys):
            ax.plot(
                xs, ys,
                color=config.color,
                linestyle='-',
                marker='.',
                markersize=4,
                label=metric_key.capitalize()
            )

    # --- Ajuste final de ejes y leyendas ---
    for ax_key, ax in axes.items():
        _draw_ap_change_annotations(ax, ap_changes)
        ax.relim()
        ax.autoscale_view()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # buscamos la configuración por la métrica
            metric = next((m for m, c in PLOT_CONFIG.items() if c.ax_key == ax_key), None)
            edge_color = PLOT_CONFIG[metric].legend_edge_color if metric else None
            ax.legend(
                handles, labels,
                loc='upper right',
                facecolor='darkslategray',
                edgecolor=edge_color,
                labelcolor='white'
            )

    ax_loss.set_ylim(bottom=-0.5, top=max(10, ax_loss.get_ylim()[1]))
    return fig


class RealTimePlot:
    """Clase para manejar la creación y actualización de la gráfica en tiempo real."""
    def __init__(
        self,
        interface: str,
        interval: float,
        log_file: Optional[str],
        target: str,
        name: Optional[str]
    ):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib no está disponible. No se puede crear la gráfica.")

        self.interface      = interface
        self.interval_sec   = interval
        self.log_file       = log_file
        self.target         = target
        self.name           = name or ""

        self.start_time     = datetime.now()
        self.start_time_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.figsize        = (24, 14)

        self.samples: List[Sample]         = []
        self.ap_changes: List[APChange]    = []
        self.current_ap: Optional[str]     = None
        self.plot_data: Dict[str, Dict]    = {}
        self.artists: List[Any]            = []

        self.iperf_client = None
        self.iperf_client = Iperf3Client(
            server_ip=self.target,
            port=5201,
            interval=self.interval_sec,
            duration=int(timedelta(hours=24).total_seconds())
        )

        self._stop_event     = threading.Event()
        self._latest_sample  = None  # última muestra recogida
        self._pending_samples = []          # <-- buffer de muestras sin procesar
        self._sample_thread  = threading.Thread(
            target=self._sampling_loop,
            daemon=True
        )

        self.collector = DataCollector(
            interface=interface,
            target_ip=target,
            interval=interval
        )

        self._initialize_plot()

    
    def _sampling_loop(self) -> None:
        """Bucle que recoge y procesa muestras cada interval_sec en un hilo aparte."""
        while not self._stop_event.is_set():
            sample = self._collect_and_process_sample()
            self._pending_samples.append(sample)
            self._latest_sample = sample
            self._log_and_print(sample)

    def _initialize_plot(self) -> None:
        self.fig = plt.figure(figsize=self.figsize)
        display = get_interface_display_name(self.interface)
        self.fig.suptitle(
            f"Monitor Wi-Fi: {display} ({self.interface}) Inicio: {self.start_time_str}",
            fontsize=16, color='white', weight='bold'
        )
        self.axes = _setup_plot_axes(self.fig)

        self.plot_data['rssi_lines'] = {}
        for mac, cfg in AP_MAP.items():
            line, = self.axes['ax_rssi'].plot(
                [], [], color=cfg['color'], linestyle='-', marker='.', markersize=4, label=cfg['name']
            )
            self.plot_data['rssi_lines'][mac] = {'line': line, 'x': [], 'y': []}

        self.plot_data['metrics'] = {}
        for key, config in PLOT_CONFIG.items():
            if key == 'rssi': continue
            ax = self.axes[config.ax_key]
            line, = ax.plot([], [], color=config.color, linestyle='-', marker='.', markersize=4, label=key.capitalize())
            self.plot_data['metrics'][key] = {'line': line, 'x': [], 'y': []}

        for ax_key, ax in self.axes.items():
            metric = next((m for m, c in PLOT_CONFIG.items() if c.ax_key == ax_key), None)
            edge_color = PLOT_CONFIG[metric].legend_edge_color if metric else None
            ax.legend(loc='upper right', facecolor='darkslategray', edgecolor=edge_color, labelcolor='white')

        self.artists = [item['line'] for item in self.plot_data['rssi_lines'].values()] + \
                       [item['line'] for item in self.plot_data['metrics'].values()]

    def _collect_and_process_sample(self) -> Sample:
        info = get_wifi_info(self.interface, self.target)
        now = datetime.now()
        elapsed = (now - self.start_time).total_seconds()
        ap_mac  = info.get('ap_mac', 'Desconocido')
        cfg     = AP_MAP.get(ap_mac, {})
        ap_name = cfg.get('name', ap_mac)

        jitter = loss = None
        if self.iperf_client and not self.iperf_client.empty():
            jitter, loss = self.iperf_client.get(block=False)

        self.current_ap = ap_mac

        sample = self.collector.sample_queue.get(timeout=1.0)
        self.samples.append(sample)
        return sample

    def _update_plot_data(self, sample: Sample) -> None:
        mac = sample.ap_mac

        # 1) Determinar AP previo
        prev_mac = None
        if len(self.samples) > 1:
            prev_mac = self.samples[-2].ap_mac

        # 2) Si ha cambiado de AP, insertar None para cortar la línea
        if prev_mac and mac != prev_mac:
            # a) cortar la línea del AP anterior (si existe)
            if prev_mac in self.plot_data['rssi_lines']:
                prev_data = self.plot_data['rssi_lines'][prev_mac]
                prev_data['x'].append(None)
                prev_data['y'].append(None)
            # b) cortar la línea del AP nuevo (si ya existía antes)
            if mac in self.plot_data['rssi_lines']:
                new_data = self.plot_data['rssi_lines'][mac]
                new_data['x'].append(None)
                new_data['y'].append(None)

        # 3) Añadir el nuevo punto de RSSI (y crear línea si es AP desconocido)
        if sample.rssi is not None:
            if mac not in self.plot_data['rssi_lines']:
                # Creamos la entrada para este AP dinámico
                line, = self.axes['ax_rssi'].plot(
                    [], [], color='gray', linestyle=':', marker='x', markersize=3,
                    label=f"AP Desc ({mac})"
                )
                self.plot_data['rssi_lines'][mac] = {'line': line, 'x': [], 'y': []}
                self.artists.append(line)
                # refrescamos leyenda
                self.axes['ax_rssi'].legend(
                    loc='upper right',
                    facecolor='darkslategray', edgecolor='gold', labelcolor='white'
                )
            data = self.plot_data['rssi_lines'][mac]
            data['x'].append(sample.elapsed)
            data['y'].append(sample.rssi)

        # 4) (Opcional) Mismo corte para latencia/jitter/pérdida
        for key, metric_data in self.plot_data['metrics'].items():
            metric_data['x'].append(sample.elapsed)
            metric_data['y'].append(getattr(sample, key, None))


    def _update_artists(self, sample: Sample) -> None:
        for data in self.plot_data['rssi_lines'].values():
            data['line'].set_data(data['x'], data['y'])
        for data in self.plot_data['metrics'].values():
            data['line'].set_data(data['x'], data['y'])

        for ax in self.axes.values():
            ax.set_xlim(0, sample.elapsed + 1)
            ax.relim()
            ax.autoscale_view(scalex=False)
            _draw_ap_change_annotations(ax, self.collector.ap_changes)

    def _log_and_print(self, sample: Sample) -> None:
        ts = sample.timestamp.strftime("%H:%M:%S.%f")[:-3]
        ts_fmt = f"[cyan]{ts:<14}[/cyan]"

        elapsed = f"{sample.elapsed:.3f} s"
        elapsed_fmt = f"[gold1]{elapsed:<12}[/gold1]"

        if len(self.samples) > 1:
            prev_elapsed = self.samples[-2].elapsed
            delta = sample.elapsed - prev_elapsed
            delta_fmt = f"{delta:.3f} s"
        else:
            delta_fmt = elapsed
        
        delta_fmt = f"[aquamarine1]{delta_fmt:<12}[/aquamarine1]"

        ap = sample.ap_name[:17]
        ap_fmt = f"[magenta]{ap:<19}[/magenta]"

        rssi_fmt = format_stat(sample.rssi, "{:.0f}", " dBm", "rssi", 12)
        lat_fmt  = format_stat(sample.latency, "{:.3f}", " ms",  "latency", 14)
        jit_fmt  = format_stat(sample.jitter, "{:.3f}", " ms",  "jitter",  14)
        loss_fmt = format_stat(sample.loss,   "{:.2f}", " %",   "loss",    10)

        row = (
            f"{ts_fmt}| "
            f"{ap_fmt}"
            f"{elapsed_fmt}"
            f"{delta_fmt}"
            f"{rssi_fmt}"
            f"{lat_fmt}"
            f"{jit_fmt}"
            f"{loss_fmt}"
        )
        console.print(row)

        if self.log_file:
            write_log_line(self.log_file, self.interface, sample)


    def _update_frame(self, frame_num: int) -> List:
        """
        Se llama cada segundo (1000 ms): procesa todas las muestras
        acumuladas y actualiza la gráfica con TODO lo nuevo.
        """
        # Si no hay nada nuevo, no hacemos nada
        if not self._pending_samples:
            return self.artists

        # Procesamos cada muestra pendiente (en orden)
        for sample in self._pending_samples:
            self._update_plot_data(sample)

        # Ya están procesadas, limpiamos el buffer
        self._pending_samples.clear()

        # Actualizamos la vista usando la última muestra recogida
        if self._latest_sample is not None:
            self._update_artists(self._latest_sample)

        return self.artists

    def start(self) -> None:
        console.print(f"\nIniciando monitorización en '{self.interface}'...")
        console.print(f"Target: {self.target}")
        #if self.iperf_client:
            #elf.iperf_client.start()

        self.collector.start()

        # Arrancamos el hilo de muestreo antes de la animación
        self._stop_event.clear()
        self._sample_thread.start()

        header = (
            f"[cyan]{'Hora':<14}[/cyan]| "
            f"[magenta]{'AP':<19}[/magenta]"
            f"[gold1]{'Tiempo':<12}[/gold1]"
            f"[aquamarine1]{'ΔTiempo':<12}[/aquamarine1]"
            f"[rssi]{'RSSI':<12}[/rssi]"
            f"[latency]{'Latencia':<14}[/latency]"
            f"[jitter]{'Jitter':<14}[/jitter]"
            f"[loss]{'Pérdida':<10}[/loss]"
        )
        console.print(header)
        console.print("-" * 104)

        self.ani = animation.FuncAnimation(
            self.fig, self._update_frame,
            interval=500,
            blit=False, cache_frame_data=False,
            save_count=0
        )
        plt.show()
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self.collector.stop()
        #if self.iperf_client:
            #self.iperf_client.stop()
        console.print("[warn]Monitorización detenida.[/warn]")

    def save_plot_image(self, new_size: Optional[Tuple[int,int]] = None) -> bool:
        fig_to_save = generate_final_plot(
            self.interface, self.samples, self.collector.ap_changes,
            self.start_time_str, new_size or self.figsize,
            self.target
        )
        if not fig_to_save:
            console.print("[error]No hay datos para generar la imagen.[/error]")
            return False

        # --- Creamos subcarpeta por interfaz ---
        base_dir = "graficas"
        iface_dir = os.path.join(base_dir, get_interface_display_name(self.interface))
        os.makedirs(iface_dir, exist_ok=True)
        
        fn = os.path.join(
            iface_dir,
            f"grafica_{self.interface}_{self.name}_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        )
        try:
            fig_to_save.savefig(fn, facecolor='darkslategray', bbox_inches='tight', dpi=150)
            console.print(f"[success]Imagen guardada en:[/] {os.path.abspath(fn)}")
            plt.close(fig_to_save)
            return True
        except Exception as e:
            console.print(f"[error]Error guardando imagen: {e}[/]")
            return False

    def save_plot_data_csv(self) -> bool:
        if not self.samples:
            console.print("[yellow]No hay muestras para guardar en CSV.[/yellow]")
            return False

        # 1) Preparamos los metadatos
        metadata = {
            'Interface': self.interface,
            'Interval_Seconds': self.interval_sec,
            'Start_Time': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'Figure_Size_Inches_Width': self.figsize[0],
            'Figure_Size_Inches_Height': self.figsize[1],
            'Ping_Target': self.target,
            'AP_Map_JSON': json.dumps(AP_MAP),
            'AP_Changes_JSON': json.dumps([
                {'time': c.time, 'name': c.name} for c in self.collector.ap_changes
            ]),
        }

        # 2) Preparamos las filas con el formato antiguo
        rows = [{
            'Timestamp':    s.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'Segundos':     s.elapsed,
            'RSSI(dBm)':    s.rssi,
            'AP_MAC':       s.ap_mac,
            'AP_Nombre':    s.ap_name,
            'Latencia(ms)': s.latency,
            'Jitter(ms)':   s.jitter,
            'Perdida(%)':   s.loss
        } for s in self.samples]

        df = pd.DataFrame(rows)

        # 3) Directorio y nombre de archivo
        base_dir = "datos_graficas"
        iface_dir = os.path.join(base_dir, get_interface_display_name(self.interface))
        os.makedirs(iface_dir, exist_ok=True)

        fn = os.path.join(
            iface_dir,
            f"datos_{self.interface}_{self.name}_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )

        try:
            # 4) Calculamos medias y las añadimos como última fila
            means = df.mean(numeric_only=True)
            mean_row = {
                'Timestamp':   'Media',          # etiqueta en columna de tiempo
                'Segundos':    '',
                'RSSI(dBm)':   means['RSSI(dBm)'],
                'AP_MAC':      '',               # puedes dejar en blanco
                'AP_Nombre':   '',
                'Latencia(ms)':means['Latencia(ms)'],
                'Jitter(ms)':  means['Jitter(ms)'],
                'Perdida(%)':  means['Perdida(%)'],
            }
            # Concatenamos la fila de medias
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

            # 5) Escribimos metadatos + tabla (incluyendo la fila de medias)
            with open(fn, 'w', newline='') as f:
                f.write("#METADATA_START\n")
                for key, val in metadata.items():
                    f.write(f"#{key},{val}\n")
                f.write("#METADATA_END\n")
                df.to_csv(f, index=False)

                console.print(f"[success]CSV guardado en:[/] {os.path.abspath(fn)}")
                return True

        except Exception as e:
            console.print(f"[error]Error guardando CSV: {e}[/]")
            return False
        
    def print_summary(self) -> None:
        """
        Muestra por consola las medias de RSSI, latencia, jitter y pérdida
        calculadas sobre self.samples en una tabla coloreada.
        """
        if not self.samples:
            console.print("[invalid]No hay datos para calcular medias.[/]")
            return

        # Calcular medias usando claves semánticas de PLOT_CONFIG
        metrics: Dict[str, Optional[float]] = {}
        for key, _ in PLOT_CONFIG.items():
            vals = [getattr(s, key) for s in self.samples if getattr(s, key) is not None]
            metrics[key] = (sum(vals) / len(vals)) if vals else None

        # Construir tabla con ylabel y unidad
        table = Table(title="Resumen de la sesión")
        table.add_column("Métrica", style="bold")
        table.add_column("Media", justify="right")
        table.add_column("Unidad")
        for key, mean in metrics.items():
            if mean is not None:
                cfg = PLOT_CONFIG[key]
                ylabel = cfg.ylabel
                # Nombre antes de paréntesis y extracción de unidad
                name = ylabel.split(' (')[0]
                unit = ''
                if '(' in ylabel and ')' in ylabel:
                    unit = ylabel[ylabel.find('(')+1:ylabel.find(')')]
                # Formatear valor con unidad y color
                colored = f"[{key}_mean]{mean:.3f}[/]"
                table.add_row(name, colored, unit)

        console.print(table)
