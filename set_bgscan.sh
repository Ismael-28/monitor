#!/bin/bash

# Script que lista las redes Wi-Fi configuradas y luego
# intenta aplicar el comando 'set_network <ID> bgscan "prio:median"'
# a cada una de ellas, según la petición del usuario.
#
# ADVERTENCIA: Según la documentación y el comportamiento estándar de wpa_cli,
# 'bgscan' es un parámetro GLOBAL para la interfaz (configurado con 'set bgscan'),
# y NO una propiedad de red individual. Este script intenta usar 'set_network'
# para 'bgscan' bajo la petición explícita del usuario, pero es muy probable
# que NO FUNCIONE como se espera para configurar 'bgscan'.
#
# Debe ejecutarse con sudo.

# --- Funciones para el modo interactivo ---

# Function to display a menu and select an interface
function select_interface() {
  echo "Buscando interfaces Wi-Fi disponibles..."
  # Get wireless interfaces and store them in an array
  local interfaces=($(iwconfig 2>/dev/null | grep -o '^[[:alnum:]]\+'))

  if [ ${#interfaces[@]} -eq 0 ]; then
    echo "Error: No se encontraron interfaces Wi-Fi. Asegúrate de que el Wi-Fi está activado."
    exit 1
  fi

  echo "Por favor, elige una interfaz:"
  PS3="Elige una opción: " # Prompt for the select menu

  select opt in "${interfaces[@]}" "Salir"; do
    case $opt in
      "Salir")
        echo "Saliendo del script."
        exit 0
        ;;
      *)
        # Verify if the chosen option is valid
        if [[ " ${interfaces[@]} " =~ " ${opt} " ]]; then
          INTERFACE="$opt"
          echo "Interfaz seleccionada: $INTERFACE"
          break
        else
          echo "Opción no válida. Inténtalo de nuevo."
        fi
        ;;
    esac
  done
}

# Función para mostrar el uso del script
mostrar_uso() {
    echo "Uso: sudo ./change_bgscan.sh <interfaz_wifi>"
    echo "       o"
    echo "     sudo ./change_bgscan.sh (para seleccionar interactivamente)"
    echo "Ejemplo: sudo ./change_bgscan.sh wlan0"
    exit 1
}

# --- Comprobaciones iniciales ---
# Verificar si el script se ejecuta con sudo
if [ "$(id -u)" -ne 0 ]; then
    echo "Este script debe ejecutarse con sudo."
    mostrar_uso
fi

interfaz=""
# Verificar si se proporcionó un argumento
if [ -z "$1" ]; then
    echo "No se proporcionó una interfaz. Iniciando selección interactiva..."
    select_interface
    interfaz="$INTERFACE" # Asignar la interfaz seleccionada a la variable 'interfaz'
else
    interfaz="$1"
fi

echo "Operando en la interfaz: $interfaz"

# Verificar si la interfaz existe y está activa
if ! ip link show "$interfaz" > /dev/null 2>&1; then
    echo "Error: La interfaz '$interfaz' no existe o no está disponible."
    echo "Interfaces Wi-Fi disponibles:"
    ip -br link show | awk '$1 ~ /^(wl|enp.*s0f|eth.*)\w+/ && ($2 == "UNKNOWN" || $2 == "UP") {print $1}' | grep -E '^(wl|enp|eth)[0-9]'
    exit 1
fi

echo "--- Listando redes Wi-Fi configuradas para '$interfaz' ---"
# Captura la salida de list_networks
networks_output=$(wpa_cli -i "$interfaz" list_networks 2>&1)

if echo "$networks_output" | grep -q "FAIL"; then
    echo "Error al listar redes para '$interfaz'. Asegúrate de que wpa_supplicant esté corriendo y gestionando la interfaz."
    echo "Detalles del error: $networks_output"
    exit 1
fi

echo "$networks_output"
echo ""

# Extraer IDs de red, ignorando la primera línea (cabecera)
# y extrayendo el primer campo (ID) de cada línea.
network_ids=$(echo "$networks_output" | tail -n +2 | awk '{print $1}')

if [ -z "$network_ids" ]; then
    echo "No se encontraron redes configuradas para '$interfaz'."
    echo "Por lo tanto, no se pueden aplicar comandos 'set_network'."
else
    echo "--- Intentando configurar 'bgscan' para cada ID de red (según lo solicitado) ---"
    for net_id in $network_ids; do
        bgscan_value='"simple:30:-120:30000"'
        echo "Intentando 'set_network $net_id bgscan \"$bgscan_value\"' para la red ID: $net_id"
        if wpa_cli -i "$interfaz" set_network "$net_id" bgscan $bgscan_value; then
            echo "  Comando 'set_network $net_id bgscan $bgscan_value' reportó ÉXITO para ID $net_id."
        else
            echo "  Comando 'set_network $net_id bgscan $bgscan_value' reportó FALLO para ID $net_id."
        fi
    done
fi

echo ""
echo "--- Verificando el estado global de 'bgscan' para la interfaz (MÉTODO ESTÁNDAR) ---"
# Intenta obtener el valor global de bgscan para la interfaz
global_bgscan_status=$(wpa_cli -i "$interfaz" get_network 0 bgscan 2>&1)

if echo "$global_bgscan_status" | grep -q "FAIL"; then
    echo "No se pudo obtener el estado global de 'bgscan'. Error: $global_bgscan_status"
else
    echo "Estado global de 'bgscan' para $interfaz: $global_bgscan_status"
fi


exit 0