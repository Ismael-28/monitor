# theme.py
from dataclasses import dataclass
from rich.theme import Theme
from rich.console import Console

@dataclass(frozen=True)
class AppTheme:
    # Colores semánticos
    info: str = "cyan"
    warn: str = "yellow"
    error: str = "bold red"
    invalid: str = "red"
    filename: str = "blue"
    success: str = "green"

    # Métricas de red
    rssi: str = "green1"
    latency: str = "cornflower_blue"
    jitter: str = "deep_pink4"
    loss: str  = "dark_orange3"

    def rich_theme(self) -> Theme:
        return Theme({
            "info":        self.info,
            "warn":        self.warn,
            "error":       self.error,
            "invalid":     self.invalid,
            "filename":    self.filename,
            "success":     self.success,
            "rssi":        self.rssi,
            "latency":     self.latency,
            "jitter":      self.jitter,
            "loss":        self.loss,
            "rssi_mean":   f"bold underline {self.rssi}",
            "latency_mean":f"bold underline {self.latency}",
            "jitter_mean": f"bold underline {self.jitter}",
            "loss_mean":   f"bold underline {self.loss}",
        })

APP_THEME = AppTheme()
console = Console(theme=APP_THEME.rich_theme())
