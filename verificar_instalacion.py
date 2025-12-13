"""
Script de verificacion de instalacion
Verifica que todas las dependencias esten correctamente instaladas
"""

import sys

print("=" * 60)
print("VERIFICACION DE INSTALACION - TP3")
print("=" * 60)
print()

# Verificar Python
print(f"[OK] Python {sys.version.split()[0]} detectado")
print()

# Verificar librerias
errores = []

print("Verificando librerias necesarias...")
print()

# NumPy
try:
    import numpy as np
    print(f"[OK] NumPy {np.__version__} instalado correctamente")
except ImportError:
    print("[ERROR] NumPy NO encontrado")
    errores.append("numpy")

# OpenCV
try:
    import cv2
    print(f"[OK] OpenCV {cv2.__version__} instalado correctamente")
except ImportError:
    print("[ERROR] OpenCV NO encontrado")
    errores.append("opencv-python")

# Matplotlib
try:
    import matplotlib
    print(f"[OK] Matplotlib {matplotlib.__version__} instalado correctamente")
except ImportError:
    print("[ERROR] Matplotlib NO encontrado")
    errores.append("matplotlib")

print()
print("=" * 60)

if len(errores) == 0:
    print("[OK] TODAS LAS DEPENDENCIAS ESTAN INSTALADAS")
    print()
    print("El proyecto esta listo para usarse.")
    print("Para ejecutar: python main.py")
else:
    print("[ERROR] FALTAN DEPENDENCIAS")
    print()
    print("Para instalar las dependencias faltantes, ejecuta:")
    print()
    print("  pip install " + " ".join(errores))
    print()
    print("O instala todas con:")
    print("  pip install -r requirements.txt")

print("=" * 60)
