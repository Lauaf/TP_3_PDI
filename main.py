import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

UMBRAL_ROJO = 70        # Threshold fallback para canal rojo
RATIO_MIN, RATIO_MAX = 0.65, 1.35  # Aspect ratio de dados
DETECT_SCALE = 0.5      # Downscaling para deteccion mas rapida

PUNTO_AREA_MIN = 0.011  # area minima de punto
PUNTO_AREA_MAX = 0.096  # area maxima de punto
PUNTO_CIRC_MIN = 0.74   # separa puntos de ruido


def mapa_rojez(frame_bgr):
    b = frame_bgr[:, :, 0].astype(np.int16)
    g = frame_bgr[:, :, 1].astype(np.int16)
    r = frame_bgr[:, :, 2].astype(np.int16)
    roj = r - np.maximum(g, b)
    return np.clip(roj, 0, 255).astype(np.uint8)


def segmentar_dados(frame_bgr):
    roj = mapa_rojez(frame_bgr)
    roj_blur = cv2.GaussianBlur(roj, (5, 5), 0)
    _, bin_auto = cv2.threshold(roj_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = float(np.mean(bin_auto == 255))
    if white_ratio < 0.001 or white_ratio > 0.35:
        r = frame_bgr[:, :, 2]
        _, bin_img = cv2.threshold(r, UMBRAL_ROJO, 255, cv2.THRESH_BINARY)
    else:
        bin_img = bin_auto

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    return bin_img


def detectar_dados_simple(frame):
    bin_img = segmentar_dados(frame)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    cands = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area_cc = int(stats[i, cv2.CC_STAT_AREA])

        if w <= 0 or h <= 0:
            continue

        ratio = h / w
        if not (RATIO_MIN < ratio < RATIO_MAX):
            continue

        area_bbox = w * h
        extent = area_cc / float(area_bbox)
        if extent < 0.14:
            continue

        #  descartar <100 (ruido) y >10000 (fondo)
        if area_bbox < 100 or area_bbox > 10000:
            continue

        cands.append((x, y, w, h, area_bbox))

    # ordenar por area
    if not cands:
        return [], bin_img

    # Si hay menos de 5 dadods, devolver vacio
    if len(cands) < 5:
        return [], bin_img

    areas = [c[4] for c in cands]
    med = np.median(areas)
    scored = []
    for c in cands:
        x, y, w, h, area_bbox = c
        rel = abs(area_bbox - med) / (med + 1e-6)
        scored.append((rel, (x, y, w, h)))

    scored.sort(key=lambda t: t[0])
    top = [t[1] for t in scored[:5]]
    top.sort(key=lambda d: d[0])
    return top, bin_img


def contar_puntos_simple(crop_bgr):
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0

    area_dado = float(h * w)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # margen adaptativo simple
    block_size = max(11, (min(w, h) // 6) | 1)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block_size, -5)

    # Margen para evitar bordes
    m = int(max(2, min(w, h) * 0.10))
    mask = np.zeros_like(bw)
    mask[m:h-m, m:w-m] = 255
    bw = cv2.bitwise_and(bw, mask)

    # Limpiar ruido menos agresivamente
    k2 = max(3, int(min(w, h) * 0.04))
    if k2 % 2 == 0:
        k2 += 1
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel2, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    a_min = area_dado * PUNTO_AREA_MIN
    a_max = area_dado * PUNTO_AREA_MAX

    pts = 0
    for c in contours:
        a = float(cv2.contourArea(c))
        if not (a_min <= a <= a_max):
            continue
        p = float(cv2.arcLength(c, True))
        if p <= 1e-6:
            continue
        circ = 4.0 * np.pi * a / (p * p)
        if circ < PUNTO_CIRC_MIN:
            continue
        pts += 1

    return pts


def procesar_video(video_path, idx, guardar_video=True):
    print(f"\n--- Video {idx}: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir video")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolucion: {w_orig}x{h_orig} @ {fps:.1f}fps ({total} frames)")

    # analisis desde 45% hasta 95% del video
    mejor = None
    paso = max(3, int(fps * 0.14))

    for frame_idx in range(int(total * 0.45), min(total, int(total * 0.95)), paso):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Escalar para deteccion rapida
        fs = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
        dados_s, bin_s = detectar_dados_simple(fs)

        if len(dados_s) != 5:
            continue

        # Contar puntos en frame original
        boxes = []
        for x, y, ww, hh in dados_s:
            X = int(round(x / DETECT_SCALE))
            Y = int(round(y / DETECT_SCALE))
            W = int(round(ww / DETECT_SCALE))
            H = int(round(hh / DETECT_SCALE))
            X = max(0, min(w_orig - 1, X))
            Y = max(0, min(h_orig - 1, Y))
            W = max(1, min(w_orig - X, W))
            H = max(1, min(h_orig - Y, H))
            boxes.append((X, Y, W, H))

        pts = []
        for X, Y, W, H in boxes:
            crop = frame[Y:Y+H, X:X+W]
            pts.append(contar_puntos_simple(crop))

        # Validar que todos tengan puntos validos
        if all(1 <= p <= 6 for p in pts):
            total_pts = sum(pts)
            if mejor is None or total_pts > mejor[3]:
                mejor = (frame.copy(), frame_idx, boxes, total_pts, pts, bin_s.copy())
                print(f"  Frame {frame_idx}: puntos={pts}, total={total_pts}")

    if mejor is None:
        print("No se encontro un frame valido")
        cap.release()
        return

    frame, frame_idx, boxes, total_pts, pts, bin_s = mejor

    print(f"\nRESULTADO: Frame #{frame_idx} -> {len(boxes)} dados | puntos={pts} | total={total_pts}")

    # Dibujar
    res = frame.copy()
    grosor = max(2, w_orig // 200)
    for (X, Y, W, H), p in zip(boxes, pts):
        cv2.rectangle(res, (X, Y), (X + W, Y + H), (255, 255, 0), thickness=grosor)
        tam = max(0.6, min(1.6, H / 35))
        gt = max(2, int(grosor * 1.2))
        cv2.putText(res, str(p), (X, Y - grosor * 2), cv2.FONT_HERSHEY_SIMPLEX, tam, (255, 255, 0), gt)

    # Visualizar
    h, w = frame.shape[:2]
    bin_big = cv2.resize(bin_s, (w, h), interpolation=cv2.INTER_NEAREST)

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Video {idx} - Frame {frame_idx}\n{len(pts)} Dados - Puntaje: {total_pts}',
                 fontsize=16, fontweight='bold')

    axs[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')

    rojo_viz = frame.copy()
    rojo_viz[:, :, 0] = 0
    rojo_viz[:, :, 1] = 0
    axs[0, 1].imshow(cv2.cvtColor(rojo_viz, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Canal Rojo')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(bin_big, cmap='gray')
    axs[1, 0].set_title('Mascara dados')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title(f'Detectados: {pts}')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # Generar video procesado si se solicita
    if guardar_video:
        # Volver al inicio del video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Reducir resolucion de salida para ahorrar espacio
        w_out = w_orig // 3
        h_out = h_orig // 3

        if not os.path.exists("datos_salida"):
            os.makedirs("datos_salida")

        out_name = os.path.join("datos_salida", f"video_{idx}_procesado.mp4")
        writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (w_out, h_out))

        # Calcular parametros para tracking de dados
        area_frame = w_orig * h_orig
        area_min_track = int(area_frame * 0.0015)
        area_max_track = int(area_frame * 0.0025)
        dist_max = int(w_orig * 0.05)

        dados_prev = []
        cont_frames = 0
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Detectar dados en escala reducida
            fs = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
            dados_s, _ = detectar_dados_simple(fs)

            # Escalar coordenadas a resolucion original
            dados_ahora = []
            for x, y, ww, hh in dados_s:
                X = int(round(x / DETECT_SCALE))
                Y = int(round(y / DETECT_SCALE))
                W = int(round(ww / DETECT_SCALE))
                H = int(round(hh / DETECT_SCALE))
                X = max(0, min(w_orig - 1, X))
                Y = max(0, min(h_orig - 1, Y))
                W = max(1, min(w_orig - X, W))
                H = max(1, min(h_orig - Y, H))
                dados_ahora.append((X, Y, W, H))

            # mantener solo dados que estan quietos
            quietos = []
            if len(dados_prev) > 0 and len(dados_ahora) > 0:
                for d_ahora in dados_ahora:
                    x1, y1 = d_ahora[0], d_ahora[1]
                    for d_prev in dados_prev:
                        x2, y2 = d_prev[0], d_prev[1]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < dist_max:
                            X, Y, W, H = d_ahora
                            crop = frame[Y:Y+H, X:X+W]
                            pts = contar_puntos_simple(crop)
                            if 1 <= pts <= 6:
                                quietos.append((X, Y, W, H, pts))
                            break

            # Dibujar dados quietos con sus puntos
            frame_dibujado = frame.copy()
            for X, Y, W, H, pts in quietos:
                cv2.rectangle(frame_dibujado, (X, Y), (X + W, Y + H), (255, 255, 0), thickness=grosor)
                tam = max(0.6, min(1.6, H / 35))
                gt = max(2, int(grosor * 1.2))
                cv2.putText(frame_dibujado, str(pts), (X, Y - grosor * 2),
                           cv2.FONT_HERSHEY_SIMPLEX, tam, (255, 255, 0), gt)

            # Actualizar dados previos cada N frames
            cont_frames += 1
            if cont_frames >= 5:
                dados_prev = dados_ahora
                cont_frames = 0

            # Escribir frame redimensionado
            out_frame = cv2.resize(frame_dibujado, (w_out, h_out))
            writer.write(out_frame)

            if frame_num % 30 == 0:
                prog = (frame_num / total) * 100
                print(f"    {prog:.1f}% ({frame_num}/{total})")

        writer.release()
        print(f"  Video guardado: {out_name}")

    cap.release()


def main():
    print("=" * 70)
    print("TP3 - Deteccion de Dados")
    print("=" * 70)

    vids = []
    if os.path.exists("videos"):
        for f in sorted(os.listdir("videos")):
            if f.lower().endswith(".mp4"):
                vids.append(os.path.join("videos", f))

    if not vids:
        print("\nNo hay videos en la carpeta 'videos'")
        return

    print(f"\nSe encontraron {len(vids)} video(s)")
    print("\nOpciones:")
    print("  1. Solo analizar y mostrar resultados")
    print("  2. Analizar y generar videos procesados (mas lento)")

    try:
        opcion = input("\nSeleccione opcion (1 o 2): ").strip()
        generar_videos = (opcion == "2")
    except:
        generar_videos = False

    if generar_videos:
        print("\n** Se generaran videos procesados en la carpeta 'datos_salida' **\n")
    else:
        print("\n** Solo se mostraran resultados (no se generaran videos) **\n")

    for i, vid in enumerate(vids, 1):
        procesar_video(vid, i, guardar_video=generar_videos)

    print("\n" + "=" * 70)
    print("Finalizado")
    print("=" * 70)


if __name__ == "__main__":
    main()
