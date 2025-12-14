import cv2
import numpy as np
import os

def calcular_params_dados(w_frame, h_frame):
    area_frame = w_frame * h_frame
    area_min = int(area_frame * 0.0015)
    area_max = int(area_frame * 0.0025)
    dist_max = int(w_frame * 0.05)
    grosor = max(2, w_frame // 200)
    return area_min, area_max, dist_max, grosor

def validar_dado(area, w, h, area_min, area_max):
    ratio = h / w
    return area_min < area < area_max and 0.7 < ratio < 1.2

def contar_puntos(recorte):
    gris = cv2.cvtColor(recorte, cv2.COLOR_RGB2GRAY)
    umbral, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=4)

    count = 0
    for i in range(1, n_comp):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ratio = h / w if w > 0 else 0

        if 60 < area < 160 and 0.6 < ratio < 1.2:
            count += 1

    return count

def draw_dado(frame, x, y, w, h, val, grosor):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), thickness=grosor)
    font_size = grosor / 10
    cv2.putText(frame, str(val), (x, y - grosor*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), 2)
    return frame

def analizar_frame_optimizado(frame, area_min, area_max):
    canal_rojo = frame[:, :, 2]
    _, binario = cv2.threshold(canal_rojo, 80, 255, cv2.THRESH_BINARY)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binario, connectivity=8)

    dados = []
    for i in range(1, n_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if validar_dado(area, w, h, area_min, area_max):
            crop = frame[y:y+h, x:x+w].copy()
            pts = contar_puntos(crop)
            dados.append((x, y, w, h, crop, pts))

    return dados, binario

def procesar_video(video_path, idx, guardar_video=False):
    print(f"\nProcesando video {idx}: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error abriendo video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  {w_orig}x{h_orig}, {fps} fps, {total} frames")

    area_min, area_max, dist_max, grosor = calcular_params_dados(w_orig, h_orig)
    print(f"  Parametros: area=[{area_min}, {area_max}], dist_max={dist_max}")

    print(f"\n  Buscando frame con 5 dados validos...")

    mejor_frame = None
    mejor_frame_num = 0
    mejor_dados = []
    encontrado = False

    frame_inicio = int(total * 0.2)
    frame_paso = 10

    for frame_num in range(frame_inicio, total, frame_paso):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        dados, _ = analizar_frame_optimizado(frame, area_min, area_max)

        if len(dados) == 5:
            puntos_dados = [d[5] for d in dados]
            if all(p > 0 for p in puntos_dados):
                print(f"    Frame {frame_num}: 5 dados validos! Puntos: {puntos_dados}")
                mejor_frame = frame
                mejor_frame_num = frame_num
                mejor_dados = dados
                encontrado = True
                break
            else:
                dados_sin_puntos = sum(1 for p in puntos_dados if p == 0)
                print(f"    Frame {frame_num}: 5 dados pero {dados_sin_puntos} sin puntos")
        elif len(dados) < 5:
            print(f"    Frame {frame_num}: Solo {len(dados)} dados")

    if not encontrado:
        print(f"\n  [ADVERTENCIA] No se encontraron 5 dados validos")
        print(f"  Usando frame al 70% del video...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.7))
        ret, frame = cap.read()
        if ret:
            mejor_frame = frame
            mejor_frame_num = int(total * 0.7)
            mejor_dados, _ = analizar_frame_optimizado(frame, area_min, area_max)

    if mejor_frame is not None and len(mejor_dados) > 0:
        print(f"\n  Resultado: Frame {mejor_frame_num} con {len(mejor_dados)} dados")

        frame_resultado = mejor_frame.copy()
        for i, dado in enumerate(mejor_dados, 1):
            x, y, w, h, crop, pts = dado
            print(f"    Dado {i} (x={x}): {pts} puntos")
            frame_resultado = draw_dado(frame_resultado, x, y, w, h, pts, grosor)

        print(f"\n  Mostrando visualizacion...")
        try:
            import matplotlib.pyplot as plt

            canal_rojo_viz = mejor_frame[:, :, 2]
            _, binario_viz = cv2.threshold(canal_rojo_viz, 80, 255, cv2.THRESH_BINARY)

            rojo_viz = mejor_frame.copy()
            rojo_viz[:, :, 0] = 0
            rojo_viz[:, :, 1] = 0

            fig, axs = plt.subplots(2, 2, figsize=(14, 11))
            fig.suptitle(f'Video {idx} - Frame {mejor_frame_num}\n{len(mejor_dados)} Dados Detectados',
                        fontsize=16, fontweight='bold')

            axs[0, 0].imshow(cv2.cvtColor(mejor_frame, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title('Original')
            axs[0, 0].axis('off')

            axs[0, 1].imshow(cv2.cvtColor(rojo_viz, cv2.COLOR_BGR2RGB))
            axs[0, 1].set_title('Canal Rojo')
            axs[0, 1].axis('off')

            axs[1, 0].imshow(binario_viz, cmap='gray')
            axs[1, 0].set_title(f'Binarizada (umbral=80)')
            axs[1, 0].axis('off')

            axs[1, 1].imshow(cv2.cvtColor(frame_resultado, cv2.COLOR_BGR2RGB))
            puntos_str = ', '.join([str(d[5]) for d in mejor_dados])
            axs[1, 1].set_title(f'Detectados: [{puntos_str}]')
            axs[1, 1].axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"    Error mostrando viz: {e}")

    if guardar_video:
        print(f"\n  Generando video procesado...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        w_out = w_orig // 3
        h_out = h_orig // 3

        if not os.path.exists("datos_salida"):
            os.makedirs("datos_salida")

        out_name = os.path.join("datos_salida", f"video_{idx}_procesado.mp4")
        writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_out, h_out))

        dados_prev = []
        cont = 0
        frame_n = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_n += 1

            canal_r = frame[:, :, 2]
            _, binario = cv2.threshold(canal_r, 80, 255, cv2.THRESH_BINARY)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binario, connectivity=8)

            dados_ahora = []
            for i in range(1, n_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                if validar_dado(area, w, h, area_min, area_max):
                    dados_ahora.append((x, y, w, h))

            quietos = []
            if len(dados_prev) > 0:
                for d_ahora in dados_ahora:
                    x1, y1 = d_ahora[0], d_ahora[1]
                    for d_prev in dados_prev:
                        x2, y2 = d_prev[0], d_prev[1]
                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if dist < dist_max:
                            x, y, w, h = d_ahora
                            crop = frame[y:y+h, x:x+w]
                            pts = contar_puntos(crop)
                            quietos.append((x, y, w, h, pts))
                            break

            frame_dibujado = frame.copy()
            for x, y, w, h, pts in quietos:
                frame_dibujado = draw_dado(frame_dibujado, x, y, w, h, pts, grosor)

            cont += 1
            if cont >= 5:
                dados_prev = dados_ahora
                cont = 0

            out_frame = cv2.resize(frame_dibujado, (w_out, h_out))
            writer.write(out_frame)

            if frame_n % 50 == 0:
                prog = (frame_n / total) * 100
                print(f"    {prog:.1f}% ({frame_n}/{total})")

        cap.release()
        writer.release()
        print(f"  Guardado: {out_name}")
    cap.release()

def main():
    print("=" * 70)
    print("TP3 - DETECCION DE DADOS")
    print("PDI - TUIA")
    print("=" * 70)

    videos = []
    if os.path.exists("videos"):
        for f in sorted(os.listdir("videos")):
            if f.endswith('.mp4'):
                videos.append(os.path.join("videos", f))

    if not videos:
        print("\nNo hay videos en la carpeta 'videos'")
        return

    print(f"\nSe van a procesar {len(videos)} video(s)\n")

    for i, vid in enumerate(videos, 1):
        procesar_video(vid, i, guardar_video=False)

    print("\n" + "=" * 70)
    print("PROCESAMIENTO COMPLETO")
    print("=" * 70)

if __name__ == "__main__":
    main()
