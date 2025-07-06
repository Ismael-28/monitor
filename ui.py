# ui.py
import os
from utils import get_interface_display_name
from plotting import RealTimePlot
from system_utils import get_wifi_interfaces_list
from theme import console

def select_interface_dialog(mode: str):
    """Muestra un diálogo para que el usuario seleccione una interfaz de red."""
    console.print(f"\nBuscando interfaces Wi-Fi en modo [info]{mode}[/]...")
    interfaces = get_wifi_interfaces_list(mode_filter=mode)

    if not interfaces:
        console.print(f"[warn]No se encontraron interfaces en modo [bold]{mode}[/][/]")
        return None

    console.print(f"Interfaces en modo [info]{mode}[/] encontradas:")
    for i, iface in enumerate(interfaces):
        display = get_interface_display_name(iface)
        # mostramos “Amigable (real)”
        console.print(f"  {i+1}) [info]{display}[/] ([dim]{iface}[/])")

    while True:
        try:
            prompt = f"Introduce el número de la interfaz (1-{len(interfaces)}): "
            choice = int(console.input(prompt))
            if 1 <= choice <= len(interfaces):
                return interfaces[choice - 1]
            else:
                console.print("[invalid]Opción no válida[/]")
        except (ValueError, KeyboardInterrupt, EOFError):
            console.print("\n[warn]Selección cancelada[/]")
            return None

def confirm_save(file_type, file_path):
    """Pregunta al usuario si desea guardar un archivo y lo elimina si la respuesta es no."""
    if not file_path or not os.path.exists(file_path):
        return

    filename = os.path.basename(file_path)

    while True:
        prompt = (
            f"¿Quieres guardar {file_type} ([info]s[/]/[invalid]n[/]): "
        )
        response = console.input(prompt).lower().strip()
        if response == 's':
            console.print(
                f"[success]{file_type.capitalize()} guardada en:[/] "
                f"{os.path.abspath(file_path)}"
            )
            break
        elif response == 'n':
            try:
                os.remove(file_path)
                console.print(
                    f"[warn]{file_type.capitalize()} eliminada:[/] "
                    f"{os.path.abspath(file_path)}"
                )
            except OSError as e:
                console.print(f"[error]Error al eliminar el archivo: {e}[/]")
            break
        else:
            console.print("[invalid]Respuesta no válida. Por favor, introduce 's' o 'n'[/]")

def get_target_dialog():
    """
    Solicita al usuario una IP o hostname para hacer ping.
    Permite dejarlo vacío para cancelar la entrada.
    """
    try:
        prompt = "Introduce el objetivo de ping ([info]IP[/] o [info]hostname[/]): "
        target = console.input(prompt).strip()
        return target if target else None
    except (KeyboardInterrupt, EOFError):
        console.print("\n[warn]Entrada cancelada[/]")
        return None
    

def save_plot_dialog(plotter: RealTimePlot) -> None:
    """
    Pregunta al usuario si quiere guardar la gráfica y los datos
    y, en caso afirmativo, los guarda.
    """
    if not plotter:
        return

    prompt = "¿Guardar gráfica y CSV? ([info]s[/]/[invalid]n[/]): "
    resp = console.input(prompt).strip().lower()
    if resp == 's':
        plotter.save_plot_image((36, 20))
        plotter.save_plot_data_csv()
    else:
        console.print("[warn]No se guardó la gráfica ni los datos.[/]")
