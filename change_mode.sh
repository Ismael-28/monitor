#!/bin/bash

# Script para configurar una interfaz Wi-Fi en modo 'managed' o 'monitor'
# y ajustar la configuración de NetworkManager correspondientemente.
#
# Se puede ejecutar de dos formas:
# 1. No interactiva: sudo ./nombre_script.sh <interfaz> <modo>
#    Ejemplo: sudo ./nombre_script.sh wlan0 monitor
#
# 2. Interactiva: sudo ./nombre_script.sh
#    El script te pedirá que elijas la interfaz y el modo.

# --- Funciones para el modo interactivo ---

# Función para mostrar un menú y seleccionar una interfaz
function select_interface() {
  echo "Buscando interfaces Wi-Fi disponibles..."
  # Obtenemos las interfaces inalámbricas y las guardamos en un array
  local interfaces=($(iwconfig 2>/dev/null | grep -o '^[[:alnum:]]\+'))

  if [ ${#interfaces[@]} -eq 0 ]; then
    echo "Error: No se encontraron interfaces Wi-Fi. Asegúrate de que el Wi-Fi está activado."
    exit 1
  fi

  echo "Por favor, elige una interfaz:"
  PS3="Elige una opción: " # Prompt para el menú select

  select opt in "${interfaces[@]}" "Salir"; do
    case $opt in
      "Salir")
        echo "Saliendo del script."
        exit 0
        ;;
      *)
        # Verificamos si la opción elegida es válida
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

# Función para mostrar un menú y seleccionar el modo
function select_mode() {
  echo
  echo "Por favor, elige el modo de operación:"
  PS3="Elige una opción: " # Prompt para el menú select

  select opt in "managed" "monitor" "Salir"; do
    case $opt in
      "managed"|"monitor")
        MODE="$opt"
        echo "Modo seleccionado: $MODE"
        break
        ;;
      "Salir")
        echo "Saliendo del script."
        exit 0
        ;;
      *)
        echo "Opción no válida. Inténtalo de nuevo."
        ;;
    esac
  done
}


# --- Comprobaciones iniciales ---
if [ "$(id -u)" -ne 0 ]; then
  echo "Este script debe ejecutarse como root (sudo)."
  exit 1
fi

# Comprobar si las herramientas necesarias están instaladas
for cmd in ifconfig iwconfig systemctl nmcli; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: El comando '$cmd' no se encuentra. Por favor, instálalo."
        echo "(net-tools para ifconfig, wireless-tools para iwconfig, systemd para systemctl, network-manager para nmcli)"
        exit 1
    fi
done


# --- Lógica principal: decidir entre modo interactivo o por argumentos ---

if [ "$#" -eq 0 ]; then
  # Modo interactivo: no se pasaron argumentos
  echo "Iniciando en modo interactivo..."
  select_interface
  select_mode
elif [ "$#" -eq 2 ]; then
  # Modo por argumentos: se pasaron dos argumentos
  echo "Argumentos detectados. Iniciando en modo no interactivo..."
  INTERFACE="$1"
  MODE="$2"
  # Validar el modo
  if [[ "$MODE" != "managed" && "$MODE" != "monitor" ]]; then
    echo "Modo inválido '$MODE'. Debe ser 'managed' o 'monitor'."
    exit 1
  fi
else
  # Número incorrecto de argumentos
  echo "Uso incorrecto."
  echo "Modo no interactivo: $0 <interfaz> <managed|monitor>"
  echo "Modo interactivo: $0 (sin argumentos)"
  exit 1
fi

# --- Ejecución de la configuración ---

echo
echo "--- Configurando interfaz '$INTERFACE' a modo '$MODE' ---"

NM_CONF_DIR="/etc/NetworkManager/conf.d"
NM_INTERFACE_UNMANAGED_FILE="${NM_CONF_DIR}/90-unmanaged-${INTERFACE}.conf"

mkdir -p "$NM_CONF_DIR"

if [ "$MODE" == "monitor" ]; then
  echo "1. Configurando NetworkManager para que NO gestione '$INTERFACE' (modo monitor)."
  echo -e "[keyfile]\nunmanaged-devices=interface-name:$INTERFACE" > "$NM_INTERFACE_UNMANAGED_FILE"
  if [ $? -eq 0 ]; then
    echo "   Archivo de configuración '$NM_INTERFACE_UNMANAGED_FILE' creado/actualizado."
  else
    echo "   Error al escribir en '$NM_INTERFACE_UNMANAGED_FILE'."
    exit 1
  fi

  echo "2. Recargando configuración de NetworkManager para aplicar la no gestión..."
  if systemctl reload NetworkManager; then
    echo "   Configuración de NetworkManager recargada."
  else
    echo "   Advertencia: No se pudo recargar NetworkManager. Intentando reiniciar el servicio..."
    if systemctl restart NetworkManager; then
      echo "   NetworkManager reiniciado."
    else
      echo "   Error: No se pudo recargar ni reiniciar NetworkManager. Los cambios pueden no aplicarse."
      exit 1 # Es crítico que NetworkManager libere el control
    fi
  fi
  sleep 4 # Dale más tiempo a NetworkManager para procesar esto y liberar el control.

  echo "3. Poniendo la interfaz '$INTERFACE' DOWN..."
  if ifconfig "$INTERFACE" down; then
    echo "   Interfaz '$INTERFACE' bajada correctamente."
  else
    echo "   Error al bajar la interfaz '$INTERFACE'. ¿Existe o está activa?"
    exit 1
  fi
  sleep 1

  echo "4. Estableciendo modo '$MODE' en '$INTERFACE' usando iwconfig..."
  if iwconfig "$INTERFACE" mode "$MODE"; then
    echo "   Modo '$MODE' establecido correctamente en '$INTERFACE'."
  else
    echo "   Error al establecer el modo '$MODE' en '$INTERFACE'."
    echo "   Asegúrate de que '$INTERFACE' es una interfaz Wi-Fi válida y soporta este modo."
    echo "   Intentando restaurar '$INTERFACE' a UP..."
    ifconfig "$INTERFACE" up &>/dev/null
    exit 1
  fi
  sleep 1

  echo "5. Poniendo la interfaz '$INTERFACE' UP..."
  if ifconfig "$INTERFACE" up; then
    echo "   Interfaz '$INTERFACE' levantada correctamente."
  else
    echo "   Error al levantar la interfaz '$INTERFACE'."
    exit 1
  fi

elif [ "$MODE" == "managed" ]; then
  echo "1. Configurando NetworkManager para que SÍ gestione '$INTERFACE' (modo managed)."
  if [ -f "$NM_INTERFACE_UNMANAGED_FILE" ]; then
    if rm -f "$NM_INTERFACE_UNMANAGED_FILE"; then
      echo "   Archivo de configuración de no gestión eliminado."
    else
      echo "   Error al eliminar '$NM_INTERFACE_UNMANAGED_FILE'."
    fi
  else
    echo "   No se encontró archivo de no gestión. NetworkManager debería gestionarla por defecto."
  fi

  echo "2. Recargando configuración de NetworkManager..."
  if systemctl reload NetworkManager; then
    echo "   Configuración de NetworkManager recargada."
  else
    echo "   Advertencia: No se pudo recargar NetworkManager. Intentando reiniciar el servicio..."
    if systemctl restart NetworkManager; then
      echo "   NetworkManager reiniciado."
    else
      echo "   Error: No se pudo recargar ni reiniciar NetworkManager. Los cambios pueden no aplicarse."
    fi
  fi
  sleep 3 # Dale un poco más de tiempo para que NetworkManager asuma el control.

  echo "3. Poniendo la interfaz '$INTERFACE' DOWN..."
  if ifconfig "$INTERFACE" down; then
    echo "   Interfaz '$INTERFACE' bajada correctamente."
  else
    echo "   Error al bajar la interfaz '$INTERFACE'. ¿Existe o está activa?"
    exit 1
  fi
  sleep 1

  echo "4. Estableciendo modo '$MODE' en '$INTERFACE' usando iwconfig..."
  if iwconfig "$INTERFACE" mode "$MODE"; then
    echo "   Modo '$MODE' establecido correctamente en '$INTERFACE'."
  else
    echo "   Error al establecer el modo '$MODE' en '$INTERFACE'."
    echo "   Asegúrate de que '$INTERFACE' es una interfaz Wi-Fi válida y soporta este modo."
    echo "   Intentando restaurar '$INTERFACE' a UP..."
    ifconfig "$INTERFACE" up &>/dev/null
    exit 1
  fi
  sleep 1

  echo "5. Poniendo la interfaz '$INTERFACE' UP..."
  if ifconfig "$INTERFACE" up; then
    echo "   Interfaz '$INTERFACE' levantada correctamente."
  else
    echo "   Error al levantar la interfaz '$INTERFACE'."
    exit 1
  fi
fi

echo
echo "--- Configuración completada ---"
echo "Estado actual de '$INTERFACE':"
iwconfig "$INTERFACE" | grep "Mode:"
echo "Estado de '$INTERFACE' según NetworkManager:"
nmcli device show "$INTERFACE" | grep "GENERAL.STATE"

exit 0