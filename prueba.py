#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para procesar múltiples ficheros CSV de monitorización Wi-Fi.
Recorre todo un directorio, agrupa ficheros por nombres similares y muestra las medias de fichero y de grupo.
Usa Rich para salida estilizada.
"""

import os
import glob
import json
import re
import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel
from rich.align import Align

console = Console()


def find_csv_files(directory: str) -> list:
    """
    Busca todos los ficheros CSV en `directory`.
    """
    pattern = os.path.join(directory, "*.csv")
    return sorted(glob.glob(pattern))


def parse_csv_file(filepath: str):
    """
    Lee el CSV con metadatos y devuelve:
      - df: DataFrame con los datos numéricos
      - events: lista de eventos con mensajes
    """
    metadata = {}
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        line = f.readline().strip()
        if line != "#METADATA_START":
            raise ValueError(f"{filepath} no tiene bloque METADATA_START")
        while True:
            line = f.readline().strip()
            if line == "#METADATA_END":
                break
            if line.startswith("#"):
                key, val = line[1:].split(',', 1)
                metadata[key] = val
        df = pd.read_csv(f)

    events = json.loads(metadata.get('Event_list_JSON', '[]'))
    return df, events


def compute_file_means(df: pd.DataFrame) -> dict:
    """
    Calcula la media de todas las columnas numéricas del DataFrame.
    """
    return df.select_dtypes(include=['number']).mean().round(3).to_dict()


def compute_durations(events: list, keyword: str) -> list:
    """
    Extrae valores de duración (en ms) de eventos cuyo mensaje contenga `keyword`.
    """
    durations = []
    pattern = re.compile(rf"{keyword}: ([\d\.]+) ms")
    for ev in events:
        msg = ev.get('msg', '')
        m = pattern.search(msg)
        if m:
            durations.append(float(m.group(1)))
    return durations


def main():
    parser = argparse.ArgumentParser(
        description="Agrupa CSV de monitorización por nombre y muestra medias individuales y globales"
    )
    parser.add_argument('directory', help='Directorio con los ficheros CSV')
    args = parser.parse_args()

    files = find_csv_files(args.directory)
    if not files:
        console.print(f"[red]No se encontraron CSV en {args.directory}[/]")
        return

    console.print(Rule("Procesando CSV en directorio", style="green"))
    records = []
    for path in files:
        filename = os.path.basename(path)
        console.print(f"- Leyendo [bold cyan]{filename}[/]")
        try:
            df, events = parse_csv_file(path)
        except Exception as e:
            console.print(f"[red]Error leyendo {filename}: {e}[/]")
            continue
        means = compute_file_means(df)
        roam = compute_durations(events, 'Tiempo reconexión')
        scan = compute_durations(events, 'Duración escaneo')
        means['Roaming(ms)'] = round(sum(roam)/len(roam),3) if roam else None
        means['Escaneo(ms)'] = round(sum(scan)/len(scan),3) if scan else None
        record = {'Archivo': filename}
        record.update(means)
        records.append(record)

    if not records:
        console.print("[yellow]No hay datos válidos para procesar.[/]")
        return

    # DataFrame con medias por fichero
    df_summary = pd.DataFrame(records)
    # Extraer nombre sin extensión
    df_summary['FilenameNoExt'] = df_summary['Archivo'].str[:-4]
    # Quitar la parte de timestamp (tras último '_')
    df_summary['Base'] = df_summary['FilenameNoExt'].str.rsplit('_', n=1).str[0]
    # Quitar el sufijo de ejecución (tras último '-') para definir grupo
    df_summary['Grupo'] = df_summary['Base'].str.rsplit('-', n=1).str[0]

    # Mostrar resultados por grupo: tabla por grupo con fila final de medias
    for grupo, subdf in df_summary.groupby('Grupo'):
        console.print(Rule(f"Grupo: {grupo}", style="magenta"))
        # Columnas a mostrar
        cols = ['Archivo'] + [c for c in subdf.columns if c not in ['Archivo','FilenameNoExt','Base','Grupo']]
        table = Table(show_header=True, header_style="bold cyan")
        for col in cols:
            justify = 'left' if col == 'Archivo' else 'right'
            table.add_column(col, justify=justify)
        # Filas por fichero
        for _, row in subdf.iterrows():
            table.add_row(*[str(row[col]) if pd.notna(row[col]) else '-' for col in cols])
        # Fila de medias del grupo
        table.add_section()
        media = subdf.drop(columns=['Archivo','FilenameNoExt','Base','Grupo']).mean(numeric_only=True).round(3)
        mean_vals = ['Media'] + [str(media[c]) for c in cols if c != 'Archivo']
        table.add_row(*mean_vals, style="bold yellow")
        # Centrar la tabla en el panel usando Align.center
        console.print(Align.center(table))

        
if __name__ == '__main__':
    main()
