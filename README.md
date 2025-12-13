# TP3 - Detección de Dados en Videos

**Trabajo Práctico 3** - Procesamiento de Imágenes
**Tecnicatura Universitaria en Inteligencia Artificial - UNR**

## Descripción

Este proyecto procesa videos de tiradas de dados para detectar automáticamente cuándo los dados están en reposo y contar los puntos de cada uno.

### Funcionalidades

- Detecta automáticamente frames donde los dados están detenidos
- Cuenta los puntos en cada dado (1-6)
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

## Estructura del Proyecto

```
TP3-Lau/
│
├── main.py                  # Programa principal
├── requirements.txt         # Dependencias
├── README.md               # Este archivo
├── MACHETE.txt             # Explicación detallada del código
│
├── videos/                 # Colocar aquí los videos a procesar
│   ├── tirada_1.mp4
│   ├── tirada_2.mp4
│   └── ...
│
└── datos_salida/           # Videos procesados (se crea automáticamente)
    ├── video_1_procesado.mp4
    ├── video_2_procesado.mp4
    └── ...
```

## Uso

1. **Colocar los videos**: Copiar los archivos `.mp4` en la carpeta `videos/`

2. **Ejecutar el programa**:
   ```bash
   python main.py
   ```

3. **Resultados**: Los videos procesados se guardarán en `datos_salida/`

## Funcionamiento Técnico

### Proceso de Detección

El algoritmo sigue estos pasos para cada frame:

1. **Filtrado de color**: Se eliminan los canales verde y azul, dejando solo el rojo (los dados son rojos)

2. **Binarización**: Se aplica un umbral al canal rojo para separar dados del fondo

3. **Componentes conectados**: Se detectan todas las regiones blancas conectadas

4. **Filtrado geométrico**: Se filtran componentes por:
   - Área (entre 3700 y 5500 píxeles)
   - Relación de aspecto (entre 0.7 y 1.2, casi cuadrados)

5. **Detección de reposo**: Se compara la posición de cada dado con frames anteriores
   - Si un dado está en la misma posición → está quieto

6. **Conteo de puntos**: Para dados quietos:
   - Se recorta la imagen del dado
   - Se detectan puntos blancos usando componentes conectados
   - Se filtran por área y forma
   - Se cuentan los componentes válidos

7. **Visualización**: Se dibuja un rectángulo amarillo y el valor sobre cada dado quieto

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
- El conteo de puntos puede fallar si el dado está muy inclinado o borroso

## Tecnologías Utilizadas

- **OpenCV**: Procesamiento de video y detección de componentes conectados
- **NumPy**: Operaciones matemáticas y manipulación de arrays
- **Python**: Lenguaje de programación

## Conceptos de PDI Aplicados

- **Umbralización (Thresholding)**: Separación de primer plano y fondo
- **Componentes Conectados**: Detección y análisis de regiones
- **Filtrado por Características Geométricas**: Área y relación de aspecto
- **Seguimiento de Objetos**: Comparación entre frames por distancia euclidiana
- **Segmentación por Color**: Filtrado de canales RGB

## Autor

Trabajo realizado para la materia Procesamiento de Imágenes I (IA 4.4)
Universidad Nacional de Rosario - TUIA

## Notas

Para entender en detalle cómo funciona cada parte del código, consultar el archivo `MACHETE.txt` que contiene explicaciones línea por línea.
