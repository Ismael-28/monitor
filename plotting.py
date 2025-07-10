import os
import json
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from queue import Empty

import pandas as pd
from rich.table import Table

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
    from theme import console
    console.print(f"[yellow]Advertencia: no se pudo importar Matplotlib ({e}). Las funciones de graficación están desactivadas.[/yellow]")
    MATPLOTLIB_AVAILABLE = False

from config import AP_MAP, PLOT_CONFIG
from data_collector import DataCollector
from utils import format_stat, get_interface_display_name, write_log_line
from models import Sample, APChange
from theme import console


# -----------------------------------------------------------------------------
# Funciones de Ayuda para Trazado
# -----------------------------------------------------------------------------

def style_axis(
    ax: Axes,
    title: str,
    ylabel: str,
    color: str,
    xlabel: Optional[str] = "Tiempo (s)",
) -> None:
    """Aplica un estilo coherente a un Axes de Matplotlib."""
    ax.set_title(title, fontsize=14, color='white')
    ax.set_ylabel(ylabel, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='y')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)

def _setup_plot_axes(fig: Figure) -> Dict[str, Axes]:
    """Crea y estila los 4 ejes a partir de PLOT_CONFIG."""
    axes_map = {
        'ax_rssi': fig.add_subplot(2, 2, 1),
        'ax_lat':  fig.add_subplot(2, 2, 2),
        'ax_jit':  fig.add_subplot(2, 2, 3),
        'ax_loss': fig.add_subplot(2, 2, 4)
    }

    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.93, bottom=0.07, hspace=0.3, wspace=0.15
    )

    for config in PLOT_CONFIG.values():
        ax = axes_map[config.ax_key]
        style_axis(ax, config.title, config.ylabel, config.color)

    axes_map['ax_loss'].set_ylim(bottom=-10, top=100)
    return axes_map


def _draw_ap_change_annotations(ax: Axes, ap_changes: List[APChange]):
    # NO borres nada aquí, porque solo dibujarás cada cambio una vez
    y_min, y_max = ax.get_ylim()
    text_y = y_min + (y_max - y_min) * 0.05
    offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01  # 1% del ancho de la gráfica
    for change in ap_changes:
        # Detectamos desconexiones (puedes cambiar la condición según tu modelo)
        is_disconnect = 'desconect' in change.name.lower()
        line_color = 'orange' if is_disconnect else 'red'
        box_color  = 'darkorange' if is_disconnect else 'red'
        if is_disconnect:
            text_x = change.time - offset
            ha = 'right'
        else:
            text_x = change.time + offset
            ha = 'left' 

        # Línea vertical
        ax.axvline(
            x=change.time,
            color=line_color,
            linestyle='--',
            linewidth=1.5,
            zorder=0
        )
        # Etiqueta con fondo diferenciado si es desconexión
        ax.text(
            text_x,
            text_y,
            change.name,
            color='white',
            rotation=90,
            ha=ha,
            va='bottom',
            fontsize=9,
            bbox=dict(
                boxstyle='round',
                fc=box_color,
                alpha=0.7,
                ec='none'
            ),
            zorder=10
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
) -> Optional[Figure]:
    """Genera una figura Matplotlib estática a partir de todos los datos recogidos."""
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

    # --- Trazado de RSSI por AP con cortes en discontinuidades ---
    rssi_data_per_ap: Dict[str, Dict[str, List]] = {}
    for s in samples:
        if s.rssi is None: continue
        mac = s.ap_mac
        rssi_data_per_ap.setdefault(mac, {'x': [], 'y': []})

    last_mac: Optional[str] = None
    for s in samples:
        if s.rssi is None: continue
        mac = s.ap_mac
        # Insertar None para cortar la línea si este AP ya tenía datos y hubo un cambio
        if last_mac is not None and mac != last_mac and rssi_data_per_ap[mac]['x']:
            rssi_data_per_ap[mac]['x'].append(None)
            rssi_data_per_ap[mac]['y'].append(None)
        rssi_data_per_ap[mac]['x'].append(s.elapsed)
        rssi_data_per_ap[mac]['y'].append(s.rssi)
        last_mac = mac

    for mac, data in rssi_data_per_ap.items():
        if not data['x']: continue
        cfg = AP_MAP.get(mac, {})
        ax_rssi.plot(
            data['x'], data['y'],
            color=cfg.get("color", "gray"),
            linestyle='-' if mac in AP_MAP else ':',
            marker='.' if mac in AP_MAP else 'x',
            markersize=4,
            label=cfg.get("name", f"AP Desc ({mac})")
        )

    # --- Trazado de Latencia, Jitter, Pérdida (con cortes en cambios de AP) ---
    for metric_key, config in PLOT_CONFIG.items():
        if metric_key == 'rssi': continue
        ax = axes[config.ax_key]
        xs, ys = [], []
        last_mac = samples[0].ap_mac if samples else None
        for s in samples:
            xs.append(s.elapsed)
            ys.append(getattr(s, metric_key))
            last_mac = s.ap_mac
        if any(v is not None for v in ys):
            ax.plot(
                xs, ys, color=config.color, linestyle='-', marker='.',
                markersize=4, label=metric_key.capitalize()
            )

    # --- Ajuste final de ejes y leyendas ---
    for ax_key, ax in axes.items():
        _draw_ap_change_annotations(ax, ap_changes)
        ax.relim()
        ax.autoscale_view()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            metric = next((m for m, c in PLOT_CONFIG.items() if c.ax_key == ax_key), None)
            edge_color = PLOT_CONFIG[metric].legend_edge_color if metric else None
            ax.legend(
                handles, labels, loc='upper right', facecolor='darkslategray',
                edgecolor=edge_color, labelcolor='white'
            )

    axes['ax_loss'].set_ylim(bottom=-10, top=100)
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

        self.interface = interface
        self.target = target
        self.name = name or ""
        self.start_time = datetime.now()
        self.start_time_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.figsize = (24, 14)

        self.samples: List[Sample] = []
        self.plot_data: Dict[str, Dict] = {}
        self.artists: List[Any] = []
        self.collector = DataCollector(interface=interface, target_ip=target, interval=interval, log_file=log_file)
        self._drawn_ap_changes = 0

        self._initialize_plot()

    def _initialize_plot(self) -> None:
        """Configura la figura y los ejes iniciales para la gráfica."""
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

    def _update_plot_data(self, sample: Sample) -> None:
        """Actualiza las estructuras de datos para el trazado con una nueva muestra."""
        mac = sample.ap_mac
        prev_mac = self.samples[-2].ap_mac if len(self.samples) > 1 else None

        # Si ha cambiado de AP, insertar None para cortar las líneas
        if prev_mac and mac != prev_mac:
            # a) Cortar la línea del AP anterior (RSSI)
            if prev_mac in self.plot_data['rssi_lines']:
                self.plot_data['rssi_lines'][prev_mac]['x'].append(None)
                self.plot_data['rssi_lines'][prev_mac]['y'].append(None)
            # b) Cortar la línea del nuevo AP si ya existía (RSSI)
            if mac in self.plot_data['rssi_lines']:
                self.plot_data['rssi_lines'][mac]['x'].append(None)
                self.plot_data['rssi_lines'][mac]['y'].append(None)

        # Añadir el nuevo punto de RSSI (y crear línea si es un AP desconocido)
        if sample.rssi is not None:
            if mac not in self.plot_data['rssi_lines']:
                line, = self.axes['ax_rssi'].plot(
                    [], [], color='gray', linestyle=':', marker='x', markersize=3,
                    label=f"AP Desc ({mac})"
                )
                self.plot_data['rssi_lines'][mac] = {'line': line, 'x': [], 'y': []}
                self.artists.append(line)
                self.axes['ax_rssi'].legend(
                    loc='upper right', facecolor='darkslategray',
                    edgecolor='gold', labelcolor='white'
                )
            data = self.plot_data['rssi_lines'][mac]
            data['x'].append(sample.elapsed)
            data['y'].append(sample.rssi)

        # Añadir nuevos puntos para latencia/jitter/pérdida
        for key, metric_data in self.plot_data['metrics'].items():
            metric_data['x'].append(sample.elapsed)
            metric_data['y'].append(getattr(sample, key, None))

    def _update_artists(self, last_sample: Sample) -> None:
        # actualizas líneas como ahora…
        for data in self.plot_data['rssi_lines'].values():
            data['line'].set_data(data['x'], data['y'])
        for data in self.plot_data['metrics'].values():
            data['line'].set_data(data['x'], data['y'])

        # reescala ejes antes de dibujar anotaciones
        for ax in self.axes.values():
            ax.set_xlim(0, last_sample.elapsed + 1)
            ax.relim()
            ax.autoscale_view(scalex=False)

        # aquí solo si hay nuevos APChange
        changes = self.collector.ap_changes
        if len(changes) > self._drawn_ap_changes:
            # tomamos solo los que aún no hemos dibujado
            new_changes = changes[self._drawn_ap_changes:]
            for ax in self.axes.values():
                _draw_ap_change_annotations(ax, new_changes)
            # actualizamos contador
            self._drawn_ap_changes = len(changes)


    def _update_frame(self, frame_num: int) -> List[Any]:
        """Función llamada por FuncAnimation para actualizar la gráfica."""
        new_samples = []
        try:
            while True:
                sample = self.collector.sample_queue.get_nowait()
                new_samples.append(sample)
        except Empty:
            pass

        if not new_samples:
            return self.artists

        for sample in new_samples:
            # Añadir a la lista principal antes de procesar para que el cálculo de delta sea correcto
            self.samples.append(sample)
            self._update_plot_data(sample)

        # Actualizar los artistas una vez con la última muestra del lote
        self._update_artists(new_samples[-1])

        return self.artists

    def start(self) -> None:
        """Inicia la monitorización y muestra la gráfica en tiempo real."""
        self.collector.start()

        self.ani = animation.FuncAnimation(
            self.fig, self._update_frame, interval=500, blit=False,
            cache_frame_data=False, save_count=0
        )
        plt.show()
        self.stop()

    def stop(self) -> None:
        """Detiene la monitorización y recoge las muestras finales pendientes."""
        # 1. Detener el hilo recolector para que no se añadan más muestras a la cola.
        self.collector.stop()
        console.print("\n[warn]Monitorización detenida.[/warn]")

        # 2. Vaciar la cola de cualquier muestra que haya quedado pendiente.
        try:
            while True:
                # Usamos get_nowait() porque sabemos que no llegarán más muestras.
                sample = self.collector.sample_queue.get_nowait()
                self.samples.append(sample)
        except Empty:
            # La cola está vacía, que es la condición de salida esperada.
            pass

        if self.samples:
            console.print(f"[info]Procesamiento finalizado. Total de muestras recogidas: {len(self.samples)}[/info]")

    def save_plot_image(self, new_size: Optional[Tuple[int,int]] = None) -> bool:
        """Guarda la gráfica final como un fichero de imagen PNG."""
        fig_to_save = generate_final_plot(
            self.interface, self.samples, self.collector.ap_changes,
            self.start_time_str, new_size or self.figsize,
        )
        if not fig_to_save:
            console.print("[error]No hay datos para generar la imagen.[/error]")
            return False

        base_dir = "graficas"
        iface_dir = os.path.join(base_dir, get_interface_display_name(self.interface))
        os.makedirs(iface_dir, exist_ok=True)
        
        filename = os.path.join(
            iface_dir,
            f"grafica_{self.interface}_{self.name}_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        )
        try:
            fig_to_save.savefig(filename, facecolor='darkslategray', bbox_inches='tight', dpi=150)
            console.print(f"[success]Imagen guardada en:[/] {os.path.abspath(filename)}")
            plt.close(fig_to_save)
            return True
        except Exception as e:
            console.print(f"[error]Error guardando imagen: {e}[/]")
            return False

    def save_plot_data_csv(self) -> bool:
        """Guarda todos los datos recogidos en un fichero CSV con metadatos."""
        if not self.samples:
            console.print("[yellow]No hay muestras para guardar en CSV.[/yellow]")
            return False

        metadata = {
            'Interface': self.interface,
            'Start_Time': self.start_time_str,
            'Ping_Target': self.target,
            'AP_Map_JSON': json.dumps(AP_MAP),
            'AP_Changes_JSON': json.dumps([
                {'time': c.time, 'name': c.name} for c in self.collector.ap_changes
            ]),
        }

        rows = [{
            'Timestamp': s.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'Segundos': s.elapsed,
            'RSSI(dBm)': s.rssi,
            'AP_MAC': s.ap_mac,
            'AP_Nombre': s.ap_name,
            'Latencia(ms)': s.latency,
            'Jitter(ms)': s.jitter,
            'Perdida(%)': s.loss
        } for s in self.samples]
        df = pd.DataFrame(rows)

        base_dir = "datos_graficas"
        iface_dir = os.path.join(base_dir, get_interface_display_name(self.interface))
        os.makedirs(iface_dir, exist_ok=True)
        filename = os.path.join(
            iface_dir,
            f"datos_{self.interface}_{self.name}_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )

        try:
            means = df.mean(numeric_only=True)
            mean_row = pd.DataFrame([{
                'Timestamp': 'Media',
                'RSSI(dBm)': means.get('RSSI(dBm)'),
                'Latencia(ms)': means.get('Latencia(ms)'),
                'Jitter(ms)': means.get('Jitter(ms)'),
                'Perdida(%)': means.get('Perdida(%)'),
            }])
            df = pd.concat([df, mean_row], ignore_index=True)

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                f.write("#METADATA_START\n")
                for key, val in metadata.items():
                    f.write(f"#{key},{val}\n")
                f.write("#METADATA_END\n")
                df.to_csv(f, index=False)

            console.print(f"[success]CSV guardado en:[/] {os.path.abspath(filename)}")
            return True
        except Exception as e:
            console.print(f"[error]Error guardando CSV: {e}[/]")
            return False
