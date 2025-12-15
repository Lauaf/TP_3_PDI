# TP3 - Detección de Dados en Videos

**Trabajo Práctico 3** - Procesamiento de Imágenes
**Tecnicatura Universitaria en Inteligencia Artificial - UNR**
**Trabajo realizado por:**
 Lautaro Florenza, Katia Otterstedt y Sebastián Palacio

## Descripción

Este proyecto procesa videos de tiradas de dados para detectar automáticamente cuándo los dados están en reposo y contar los puntos de cada uno.

### Funcionalidades

- Detecta automáticamente frames donde los dados están detenidos
- Cuenta los puntos en cada dado
- Genera videos procesados con:
  - Bounding boxes alrededor de cada dado
  - Valor detectado sobre cada dado
  - Reducción de escala para archivos más livianos

## Instalación

### Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

Las librerías necesarias son:
- `numpy`: Para operaciones con arrays y matrices
- `opencv-python`: Para procesamiento de imágenes y video

## Uso

1. **Colocar los videos**: Copiar los archivos `.mp4` en la carpeta `videos/`

2. **Ejecutar el programa**:
   ```bash
   python main.py
   ```

3. **Resultados**: Los videos procesados se guardarán en `datos_salida/`


### Parámetros Configurables

En el código (`main.py`) se pueden ajustar:

- `UMBRAL_ROJO`: Sensibilidad para detectar color rojo (default: 80)
- `AREA_MIN_DADO` / `AREA_MAX_DADO`: Rango de tamaño de dados
- `FRAMES_ENTRE_COMPARACIONES`: Cada cuántos frames comparar posiciones (default: 19)
- `DISTANCIA_MAX_MOVIMIENTO`: Cuánto puede moverse un dado para considerarlo quieto (default: 80 píxeles)
- `ESCALA_VIDEO`: Factor de reducción del video de salida (default: 3)

## Limitaciones Conocidas

- Los dados deben ser rojos sobre fondo claro/verde
- Requiere iluminación razonablemente uniforme
- Funciona mejor con videos estabilizados

## Tecnologías Utilizadas

- **OpenCV**: Procesamiento de video y detección de componentes conectados
- **NumPy**: Operaciones matemáticas y manipulación de arrays
- **Python**: Lenguaje de programación



