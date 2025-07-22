import subprocess
import signal
from datetime import datetime
from multiprocessing import Process
import time
from rich.console import Console # Usamos rich directamente para el ejemplo

# --- Configuración de la consola de rich ---
# Si tienes tu propio módulo 'theme', puedes seguir usándolo.
# Para que este script sea autoejecutable, defino la consola aquí.
console = Console(highlight=False)

class APEventCollector(Process):
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
            if message:
                console.print(
                    f"[{style}]"
                    f"[dim]+{elapsed_seconds:08.3f}s[/dim] " # Timestamp del programa
                    f"[{ts_iw:%H:%M:%S.%f}] " # Timestamp de 'iw'
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


if __name__ == "__main__":
    # IMPORTANTE: Reemplaza "wlp0s20f3" con el nombre de tu interfaz Wi-Fi
    INTERFACE_WIFI = "wlp0s20f3" 
    
    collector = APEventCollector(INTERFACE_WIFI)
    console.print(f"[bold]Iniciando monitor de eventos para la interfaz [magenta]{INTERFACE_WIFI}[/magenta]...[/bold]")
    console.print("[dim]Presiona Ctrl+C para detener.[/dim]")
    collector.start()
    
    try:
        # Mantenemos el programa principal vivo mientras el proceso hijo trabaja
        while collector.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        collector.stop()
    
    console.print("[bold]Programa finalizado.[/bold]")