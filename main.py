"""
Trabajo Práctico 3 - Procesamiento de Imágenes
Detección de dados en videos usando componentes conectados

Este script procesa videos de tiradas de dados y:
- Detecta automáticamente cuando los dados están quietos
- Cuenta los puntos en cada dado
- Genera videos con bounding boxes y valores detectados
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Rutas de archivos
CARPETA_VIDEOS = "videos"
CARPETA_SALIDA = "datos_salida"

# Parámetros de detección
UMBRAL_ROJO = 80              # Umbral para binarizar canal rojo
AREA_MIN_DADO = 3700          # Área mínima de un dado en píxeles
AREA_MAX_DADO = 5500          # Área máxima de un dado en píxeles
ASPECT_MIN = 0.7              # Relación de aspecto mínima (h/w)
ASPECT_MAX = 1.2              # Relación de aspecto máxima

# Parámetros para puntos del dado
UMBRAL_PUNTOS = 120           # Umbral para detectar puntos blancos
AREA_MIN_PUNTO = 60           # Área mínima de un punto
AREA_MAX_PUNTO = 160          # Área máxima de un punto
ASPECT_MIN_PUNTO = 0.6        # Aspecto mínimo de un punto
ASPECT_MAX_PUNTO = 1.2        # Aspecto máximo de un punto

# Parámetros de movimiento
FRAMES_ENTRE_COMPARACIONES = 19    # Cada cuántos frames comparar posiciones
DISTANCIA_MAX_MOVIMIENTO = 80      # Distancia máxima para considerar mismo dado

# Parámetros de visualización
ESCALA_VIDEO = 3              # Factor de reducción del video de salida
FRAME_MUESTRA = 80            # Frame donde mostrar pasos intermedios


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_carpeta_salida():
    """Crea la carpeta de salida si no existe"""
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"[INFO] Carpeta '{CARPETA_SALIDA}' creada")


def obtener_videos():
    """
    Obtiene la lista de videos a procesar
    Retorna: lista de rutas a los videos
    """
    videos = []
    if os.path.exists(CARPETA_VIDEOS):
        for archivo in sorted(os.listdir(CARPETA_VIDEOS)):
            if archivo.endswith('.mp4'):
                ruta = os.path.join(CARPETA_VIDEOS, archivo)
                videos.append(ruta)

    if len(videos) == 0:
        print(f"[ADVERTENCIA] No se encontraron videos en '{CARPETA_VIDEOS}'")
    else:
        print(f"[INFO] Se encontraron {len(videos)} videos")

    return videos


def filtrar_canal_rojo(frame):
    """
    Satura los canales verde y azul, dejando solo el rojo

    Por qué: Los dados son rojos sobre fondo verde/celeste.
    Al eliminar verde y azul, los dados se destacan mucho más.

    Args:
        frame: imagen BGR de OpenCV

    Returns:
        frame_rojo: imagen con solo canal rojo
    """
    frame_rojo = frame.copy()
    frame_rojo[:, :, 0] = 0  # Canal azul a 0
    frame_rojo[:, :, 1] = 0  # Canal verde a 0
    # frame_rojo[:, :, 2] queda igual (canal rojo)

    return frame_rojo


def binarizar_imagen(imagen_gris, umbral):
    """
    Convierte una imagen en escala de grises a binaria

    Píxeles > umbral → 255 (blanco)
    Píxeles <= umbral → 0 (negro)

    Args:
        imagen_gris: imagen en escala de grises
        umbral: valor de corte

    Returns:
        imagen_binaria: imagen con solo valores 0 y 255
    """
    _, img_bin = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    return img_bin


def es_dado_valido(area, ancho, alto):
    """
    Determina si un componente conectado es un dado basándose en su geometría

    Criterios:
    - Área entre 3700 y 5500 píxeles (tamaño apropiado)
    - Relación de aspecto cercana a 1 (casi cuadrado)

    Args:
        area: área del componente en píxeles
        ancho: ancho del componente
        alto: alto del componente

    Returns:
        True si cumple criterios de dado, False en caso contrario
    """
    relacion_aspecto = alto / ancho

    cumple_area = AREA_MIN_DADO < area < AREA_MAX_DADO
    cumple_aspecto = ASPECT_MIN < relacion_aspecto < ASPECT_MAX

    return cumple_area and cumple_aspecto


def contar_puntos_dado(recorte_dado):
    """
    Cuenta los puntos blancos en la cara de un dado

    Proceso:
    1. Convertir a escala de grises
    2. Binarizar para resaltar puntos blancos
    3. Encontrar componentes conectados
    4. Filtrar por área y relación de aspecto
    5. Contar componentes válidos

    Args:
        recorte_dado: imagen RGB del dado recortado

    Returns:
        cantidad de puntos detectados (1-6)
    """
    # Convertir a escala de grises
    gris = cv2.cvtColor(recorte_dado, cv2.COLOR_RGB2GRAY)

    # Binarizar para resaltar puntos blancos
    binaria = binarizar_imagen(gris, UMBRAL_PUNTOS)

    # Detectar componentes conectados
    num_componentes, etiquetas, estadisticas, centroides = \
        cv2.connectedComponentsWithStats(binaria, connectivity=4)

    puntos_validos = 0

    # Revisar cada componente (empezar en 1 para saltar el fondo)
    for i in range(1, num_componentes):
        area = estadisticas[i, cv2.CC_STAT_AREA]
        ancho = estadisticas[i, cv2.CC_STAT_WIDTH]
        alto = estadisticas[i, cv2.CC_STAT_HEIGHT]

        # Calcular relación de aspecto
        relacion = alto / ancho if ancho > 0 else 0

        # Verificar si es un punto válido
        if (AREA_MIN_PUNTO < area < AREA_MAX_PUNTO and
            ASPECT_MIN_PUNTO < relacion < ASPECT_MAX_PUNTO):
            puntos_validos += 1

    return puntos_validos


def calcular_distancia(x1, y1, x2, y2):
    """
    Calcula la distancia euclidiana entre dos puntos

    Fórmula: d = √[(x1-x2)² + (y1-y2)²]

    Por qué: Para determinar si un dado se movió entre frames.
    Si la distancia es pequeña, es el mismo dado en reposo.
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def dibujar_dado(frame, x, y, ancho, alto, valor):
    """
    Dibuja un rectángulo y el valor del dado en el frame

    Args:
        frame: imagen donde dibujar
        x, y: coordenadas superior izquierda
        ancho, alto: dimensiones del rectángulo
        valor: número de puntos del dado

    Returns:
        frame con el dado dibujado
    """
    # Dibujar rectángulo amarillo
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto),
                  (255, 255, 0), thickness=10)

    # Escribir el valor arriba del dado
    cv2.putText(frame, str(valor), (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    return frame


def misma_posicion(dado_actual, dado_previo):
    """
    Verifica si dos dados están en la misma posición

    Args:
        dado_actual: tupla (x, y, w, h, recorte)
        dado_previo: tupla (x, y, w, h, recorte)

    Returns:
        True si están en la misma posición (distancia < umbral)
    """
    x1, y1 = dado_actual[0], dado_actual[1]
    x2, y2 = dado_previo[0], dado_previo[1]

    distancia = calcular_distancia(x1, y1, x2, y2)

    return distancia < DISTANCIA_MAX_MOVIMIENTO


# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ============================================================================

def procesar_video(ruta_video, indice):
    """
    Procesa un video completo, detectando dados y generando salida

    Pasos:
    1. Abrir video y obtener propiedades
    2. Crear video de salida
    3. Para cada frame:
       a. Filtrar canal rojo
       b. Binarizar
       c. Detectar componentes conectados
       d. Filtrar dados válidos
       e. Comparar con frame anterior
       f. Dibujar dados quietos
       g. Escribir frame procesado

    Args:
        ruta_video: path del video a procesar
        indice: número del video (para nombrar salida)
    """
    print(f"\n[PROCESANDO] Video {indice}: {ruta_video}")

    # Abrir video
    captura = cv2.VideoCapture(ruta_video)
    if not captura.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {ruta_video}")
        return

    # Obtener propiedades del video
    fps = int(captura.get(cv2.CAP_PROP_FPS))
    ancho_original = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_original = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  - Resolución: {ancho_original}x{alto_original}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")

    # Calcular dimensiones del video de salida
    ancho_salida = ancho_original // ESCALA_VIDEO
    alto_salida = alto_original // ESCALA_VIDEO

    # Crear video de salida
    nombre_salida = os.path.join(CARPETA_SALIDA, f"video_{indice}_procesado.mp4")
    escritor = cv2.VideoWriter(nombre_salida,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (ancho_salida, alto_salida))

    # Variables para seguimiento
    dados_previos = []
    contador_frames = 0
    frame_actual = 0
    dados_detectados_previo = set()  # Para no repetir detecciones
    mostrar_visualizacion = True  # Flag para mostrar pasos intermedios

    print("  - Procesando frames...")

    # Procesar cada frame
    while captura.isOpened():
        ret, frame = captura.read()
        if not ret:
            break

        frame_actual += 1

        # Copiar frame original para dibujar
        frame_original = frame.copy()

        # PASO 1: Filtrar canal rojo
        frame_rojo = filtrar_canal_rojo(frame)

        # PASO 2: Binarizar el canal rojo
        canal_rojo = frame_rojo[:, :, 2]  # Extraer solo canal rojo
        frame_binario = binarizar_imagen(canal_rojo, UMBRAL_ROJO)

        # PASO 3: Detectar componentes conectados
        num_labels, etiquetas, stats, centroides = \
            cv2.connectedComponentsWithStats(frame_binario, connectivity=8)

        # PASO 4: Filtrar dados válidos
        dados_actuales = []

        for i in range(1, num_labels):  # Saltar el fondo (índice 0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Verificar si es un dado
            if es_dado_valido(area, w, h):
                # Recortar el dado del frame original
                recorte = frame_original[y:y+h, x:x+w]
                dados_actuales.append((x, y, w, h, recorte))

        # PASO 5: Comparar con dados previos y dibujar si están quietos
        dados_quietos = []  # Lista de dados quietos en este frame

        for dado_actual in dados_actuales:
            for dado_previo in dados_previos:
                if misma_posicion(dado_actual, dado_previo):
                    # El dado está quieto, contamos sus puntos
                    x, y, w, h, recorte = dado_actual
                    puntos = contar_puntos_dado(recorte)

                    # Guardar para mostrar por terminal
                    dados_quietos.append((x, y, puntos))

                    # Dibujar en el frame
                    frame_original = dibujar_dado(frame_original, x, y, w, h, puntos)
                    break  # Ya encontramos match, pasar al siguiente dado

        # Mostrar detecciones por terminal cuando hay dados quietos
        if dados_quietos and len(dados_quietos) == 5:  # Los 5 dados quietos
            # Crear identificador único para este set de dados
            valores_ordenados = tuple(sorted([p for _, _, p in dados_quietos]))

            # Solo mostrar si es diferente al anterior
            if valores_ordenados != dados_detectados_previo:
                dados_detectados_previo = valores_ordenados
                print(f"\n  [FRAME {frame_actual}] Dados detenidos detectados:")
                for idx, (x, y, valor) in enumerate(sorted(dados_quietos, key=lambda d: d[0]), 1):
                    print(f"    Dado #{idx} (posicion x={x}): {valor} puntos")
                print(f"    Total: {sum([p for _, _, p in dados_quietos])} puntos")

        # Visualización de pasos intermedios (cuando detecta 5 dados quietos)
        if mostrar_visualizacion and len(dados_quietos) == 5:
            print(f"\n  [VISUALIZACION] Mostrando pasos de procesamiento (Video {indice}, Frame {frame_actual})...")
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(14, 11))
                titulo = f'Video {indice} - Pasos de Procesamiento - Frame {frame_actual}\n5 Dados Detectados en Reposo'
                fig.suptitle(titulo, fontsize=16, fontweight='bold')

                # Imagen original
                axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                axes[0, 0].set_title('1. Imagen Original', fontsize=12)
                axes[0, 0].axis('off')

                # Canal rojo filtrado
                axes[0, 1].imshow(cv2.cvtColor(frame_rojo, cv2.COLOR_BGR2RGB))
                axes[0, 1].set_title('2. Solo Canal Rojo (Verde y Azul = 0)', fontsize=12)
                axes[0, 1].axis('off')

                # Imagen binarizada
                axes[1, 0].imshow(frame_binario, cmap='gray')
                axes[1, 0].set_title('3. Binarizada (Umbral = 80)', fontsize=12)
                axes[1, 0].axis('off')

                # Resultado final con detecciones
                axes[1, 1].imshow(cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB))
                valores_str = ', '.join([str(p) for _, _, p in sorted(dados_quietos, key=lambda d: d[0])])
                axes[1, 1].set_title(f'4. Dados Detectados\nValores: [{valores_str}]', fontsize=12)
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.show()
                print(f"  [VISUALIZACION] Ventana mostrada. Cierra la ventana para continuar...")

                mostrar_visualizacion = False  # Mostrar solo una vez por video
            except ImportError:
                print("    (matplotlib no disponible para visualizacion)")
            except Exception as e:
                print(f"    (error en visualizacion: {e})")

        # PASO 6: Actualizar dados previos cada N frames
        contador_frames += 1
        if contador_frames >= FRAMES_ENTRE_COMPARACIONES:
            dados_previos = dados_actuales
            contador_frames = 0

        # PASO 7: Escalar y escribir frame
        frame_escalado = cv2.resize(frame_original,
                                    (ancho_salida, alto_salida))
        escritor.write(frame_escalado)

        # Mostrar progreso
        if frame_actual % 50 == 0:
            progreso = (frame_actual / total_frames) * 100
            print(f"    Progreso: {progreso:.1f}% ({frame_actual}/{total_frames})")

    # Liberar recursos
    captura.release()
    escritor.release()

    print(f"  - Video procesado guardado en: {nombre_salida}")


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que coordina todo el procesamiento
    """
    print("=" * 70)
    print("TRABAJO PRÁCTICO 3 - DETECCIÓN DE DADOS EN VIDEOS")
    print("Procesamiento de Imágenes - TUIA")
    print("=" * 70)

    # Crear carpeta de salida
    crear_carpeta_salida()

    # Obtener lista de videos
    videos = obtener_videos()

    if len(videos) == 0:
        print("\n[ERROR] No hay videos para procesar")
        print(f"Coloque los videos en la carpeta '{CARPETA_VIDEOS}'")
        return

    # Procesar cada video
    print(f"\nSe procesarán {len(videos)} video(s)")

    for idx, video in enumerate(videos, start=1):
        procesar_video(video, idx)

    print("\n" + "=" * 70)
    print("PROCESAMIENTO COMPLETADO")
    print(f"Los videos procesados están en la carpeta '{CARPETA_SALIDA}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
