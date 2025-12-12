import cv2
import time
from mtcnn import MTCNN
import psutil
import numpy as np


CAM_INDEX = 0
SHOW_METRICS = True # Metricas en pantalla
WINDOW_NAME = "MTCNN - Experimento en tiempo real"

# Umbral de distancia para decidir si es la misma persona
RECOG_DIST_THRESHOLD = 0.25


def seleccionar_rostro_principal(faces, frame_width, frame_height):
    """
    Selecciona el rostro "principal" entre varios:
    - Preferimos el más grande y más centrado.
    """
    if not faces:
        return None

    cx_frame, cy_frame = frame_width / 2, frame_height / 2

    best_face = None
    best_score = -1e9

    for f in faces:
        x, y, w, h = f["box"]
        # MTCNN a veces devuelve coords negativas; las arreglamos en el dibujo
        area = w * h
        cx = x + w / 2
        cy = y + h / 2
        dist_center = ((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5

        # Score simple: grande y centrado
        score = area - 0.1 * dist_center
        if score > best_score:
            best_score = score
            best_face = f

    return best_face


def compute_mtcnn_descriptor(keypoints):
    """
    Descriptor geométrico simple usando los 5 puntos de MTCNN:
      - left_eye, right_eye, nose, mouth_left, mouth_right.
      - Origen: punto medio entre los ojos.
      - Escala: distancia entre los ojos.
      - Rotación: ojos alineados horizontalmente.
      - Devuelve vector normalizado (norma 1).
    """
    names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    pts = []
    for name in names:
        if name in keypoints and keypoints[name] is not None:
            pts.append(keypoints[name])

    if len(pts) < 2:
        return None

    coords = np.array(pts, dtype=np.float32)  # (N, 2)

    # Ojos
    if "left_eye" not in keypoints or "right_eye" not in keypoints:
        return None

    left = np.array(keypoints["left_eye"], dtype=np.float32)
    right = np.array(keypoints["right_eye"], dtype=np.float32)

    # Punto medio entre ojos -> origen
    mid_eyes = (left + right) / 2.0
    coords -= mid_eyes

    # Escala: distancia entre ojos
    eye_vec = right - left
    eye_dist = np.linalg.norm(eye_vec) + 1e-6
    coords /= eye_dist

    # Rotación: alinear ojos con eje X
    dx, dy = eye_vec
    angle = np.arctan2(dy, dx)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)
    coords = coords @ R.T

    # Centrado ligero
    coords -= coords.mean(axis=0, keepdims=True)

    # Vector 1D normalizado
    desc = coords.flatten().astype(np.float32)
    norm = np.linalg.norm(desc) + 1e-6
    desc /= norm

    return desc


def main():

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    # Fijar tamaño
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = MTCNN()
    process = psutil.Process()
    frame_count = 0

    # tiempos
    total_proc_time = 0.0      # BGR2RGB + detect_faces + dibujo básico
    total_det_time = 0.0       # SOLO detect_faces

    # detección
    frames_con_rostro = 0
    frames_sin_rostro = 0
    frames_multiples_rostros = 0

    # reconocimiento
    frames_reconocido = 0
    frames_no_reconocido = 0
    distances_all = []         # estadísticas de distancias

    # ángulo (roll)
    frames_con_angulo = 0
    sum_angle_deg = 0.0
    sum_abs_angle_deg = 0.0
    max_abs_angle_deg = 0.0

    # CPU / MEM
    sum_cpu = 0.0
    sum_mem = 0.0
    max_cpu = 0.0
    max_mem = 0.0

    # Robustez temporal
    time_recognized = 0.0       # tiempo con rostro presente y reconocido
    time_not_recognized = 0.0   # tiempo con rostro presente y NO reconocido

    # Referencia
    ref_desc = None
    ref_name = ""
    ref_img_path = None
    have_ref = False

    current_desc = None
    current_face_bgr = None

    t_global_start = time.perf_counter()

    print("[INFO] Modo REGISTRO: coloca tu rostro frontalmente y pulsa 'c' para capturar referencia.")
    print("[INFO] Presiona 'q' para salir del experimento.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No se pudo leer un frame de la cámara.")
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        # Medir tiempo total por frame (aprox para FPS)
        t_frame_start = time.perf_counter()

        # Tiempo de procesamiento "core" (conversión + detección)
        t_proc_start = time.perf_counter()

        # Convertir BGR (OpenCV) a RGB (MTCNN)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Medir SOLO tiempo de detección MTCNN
        t_det_start = time.perf_counter()
        faces = detector.detect_faces(rgb)
        t_det_end = time.perf_counter()

        det_time = t_det_end - t_det_start
        total_det_time += det_time

        t_proc_end = time.perf_counter()
        total_proc_time += (t_proc_end - t_proc_start)

        # Actualizar métricas de detección
        if len(faces) == 0:
            frames_sin_rostro += 1
        else:
            frames_con_rostro += 1
            if len(faces) > 1:
                frames_multiples_rostros += 1

        # Seleccionamos un solo rostro principal
        main_face = seleccionar_rostro_principal(faces, frame_w, frame_h)

        # Estado de reconocimiento para este frame
        recognized = False
        dist = None
        score = 0.0

        current_desc = None
        current_face_bgr = None

        # Dibujar resultados
        angle_deg = None  # <--- para guardar el ángulo del frame actual

        if main_face is not None:
            x, y, w, h = main_face["box"]

            # Corregir posibles coords negativas
            x = max(0, x)
            y = max(0, y)
            w = max(0, w)
            h = max(0, h)

            # Bounding box
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0),
                          2)

            # Recorte de la cara en BGR (para guardar si se pulsa 'c')
            current_face_bgr = frame[y:y + h, x:x + w].copy()

            # Puntos clave
            keypoints = main_face.get("keypoints", {})

            for name, (kx, ky) in keypoints.items():
                cv2.circle(frame, (kx, ky), 2, (0, 0, 255), -1)
                cv2.putText(frame, name, (kx + 2, ky - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # calcular angulo de inclinación (roll)
            left_eye = keypoints.get("left_eye")
            right_eye = keypoints.get("right_eye")

            if left_eye is not None and right_eye is not None:
                (lx, ly) = left_eye
                (rx, ry) = right_eye

                dx = rx - lx
                dy = ry - ly

                angle_rad = np.arctan2(dy, dx)
                angle_deg = angle_rad * 180.0 / np.pi

                # Acumular estadísticas de ángulo
                frames_con_angulo += 1
                sum_angle_deg += angle_deg
                sum_abs_angle_deg += abs(angle_deg)
                max_abs_angle_deg = max(max_abs_angle_deg, abs(angle_deg))

                # Mostrar el ángulo cerca del bounding box
                text_angle = f"Angulo (roll): {angle_deg:+5.1f} deg"
                cv2.putText(frame, text_angle,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2)

            # Descriptor MTCNN + reconocimiento
            desc_cur = compute_mtcnn_descriptor(keypoints)
            current_desc = desc_cur

            if have_ref and ref_desc is not None and desc_cur is not None:
                dist = float(np.linalg.norm(desc_cur - ref_desc))
                distances_all.append(dist)

                score = 1.0 / (1.0 + dist)

                if dist <= RECOG_DIST_THRESHOLD:
                    recognized = True
                    frames_reconocido += 1
                else:
                    recognized = False
                    frames_no_reconocido += 1

                color = (0, 255, 0) if recognized else (0, 0, 255)
                text_score = f"dist: {dist:.3f}  sim: {score:.3f}"

                if recognized:
                    if ref_name:
                        text_label = f"Reconocido ({ref_name})"
                    else:
                        text_label = "Reconocido"
                else:
                    if ref_name:
                        text_label = f"No es ({ref_name})"
                    else:
                        text_label = "No reconocido"

                cv2.putText(frame, text_score,
                            (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
                cv2.putText(frame, text_label,
                            (x, y + h + 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        # Métricas en tiempo real
        t_frame_end = time.perf_counter()
        frame_time = t_frame_end - t_frame_start
        fps_instant = 1.0 / frame_time if frame_time > 0 else 0.0

        # Robustez temporal
        if have_ref and main_face is not None and current_desc is not None:
            if recognized:
                time_recognized += frame_time
            else:
                time_not_recognized += frame_time

        if SHOW_METRICS:
            avg_det_time_ms = (total_det_time / frame_count) * 1000.0
            avg_proc_time_ms = (total_proc_time / frame_count) * 1000.0
            fps_global = frame_count / (t_frame_end - t_global_start)

            mode_str = "RECONOCIMIENTO" if have_ref else "REGISTRO"
            text_mode = f"Modo: {mode_str}"

            cv2.putText(frame, text_mode, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            text1 = f"FPS inst: {fps_instant:5.1f}"
            text2 = f"FPS global: {fps_global:5.1f}"
            text3 = f"T_det_prom: {avg_det_time_ms:6.2f} ms"
            text3b = f"T_proc_prom: {avg_proc_time_ms:6.2f} ms"

            cv2.putText(frame, text1, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text2, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text3, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text3b, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Uso de CPU y memoria del proceso actual
            cpu_percent = process.cpu_percent(interval=0)  # 0 = no bloquear
            mem_mb = process.memory_info().rss / (1024 * 1024)

            sum_cpu += cpu_percent
            sum_mem += mem_mb
            max_cpu = max(max_cpu, cpu_percent)
            max_mem = max(max_mem, mem_mb)

            text4 = f"CPU proc: {cpu_percent:5.1f}%"
            text5 = f"MEM proc: {mem_mb:6.1f} MB"

            cv2.putText(frame, text4, (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, text5, (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if have_ref and ref_name:
                cv2.putText(frame,
                            f"Ref: {ref_name}",
                            (10, 195),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            if not have_ref:
                cv2.putText(frame,
                            "Pulsa 'c' para capturar referencia",
                            (10, frame_h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Mostrar frame
        cv2.imshow(WINDOW_NAME, frame)

        # Salir con 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Capturar referencia con 'c'
        if not have_ref and key == ord('c') and current_desc is not None:
            ref_desc = current_desc.copy()

            try:
                nombre = input("Nombre de la persona de referencia: ").strip()
            except EOFError:
                nombre = ""

            if nombre == "":
                nombre = "Persona"

            ref_name = nombre

            if current_face_bgr is not None:
                safe_name = ref_name.replace(" ", "_")
                ref_img_path = f"ref_mtcnn_{safe_name}.png"
                cv2.imwrite(ref_img_path, current_face_bgr)
                print(f"[INFO] Imagen de referencia guardada en: {ref_img_path}")

            have_ref = True
            print(f"[INFO] Descriptor de referencia capturado para {ref_name}.")


    t_global_end = time.perf_counter()
    total_time = t_global_end - t_global_start

    cap.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print("[INFO] No se procesaron frames.")
        return

    avg_det_time_ms = (total_det_time / frame_count) * 1000.0
    avg_proc_time_ms = (total_proc_time / frame_count) * 1000.0
    fps_global = frame_count / total_time

    cpu_avg = sum_cpu / frame_count if frame_count > 0 else 0.0
    mem_avg = sum_mem / frame_count if frame_count > 0 else 0.0

    if frames_con_angulo > 0:
        mean_angle = sum_angle_deg / frames_con_angulo
        mean_abs_angle = sum_abs_angle_deg / frames_con_angulo
    else:
        mean_angle = 0.0
        mean_abs_angle = 0.0

    print("\n===== RESULTADOS DEL EXPERIMENTO MTCNN =====")
    print(f"Frames totales procesados:          {frame_count}")
    print(f"Tiempo total:                       {total_time:.2f} s")
    print(f"FPS global promedio:                {fps_global:.2f}")
    print(f"Tiempo deteccion promedio:          {avg_det_time_ms:.2f} ms")
    print(f"Tiempo proc (RGB+detec) promedio:   {avg_proc_time_ms:.2f} ms")

    print("\n--- Detección de rostros (aprox) ---")
    print(f"Frames con al menos 1 rostro:       {frames_con_rostro} "
          f"({frames_con_rostro / frame_count * 100:.1f}%)")
    print(f"Frames sin rostro:                  {frames_sin_rostro} "
          f"({frames_sin_rostro / frame_count * 100:.1f}%)")
    print(f"Frames con múltiples rostros:       {frames_multiples_rostros} "
          f"({frames_multiples_rostros / frame_count * 100:.1f}%)")

    if have_ref:
        total_eval = frames_reconocido + frames_no_reconocido
        acc = (frames_reconocido / total_eval * 100.0) if total_eval > 0 else 0.0

        print("\n--- Reconocimiento (descriptor MTCNN) ---")
        print(f"Persona de referencia:              {ref_name}")
        if ref_img_path:
            print(f"Imagen de referencia:               {ref_img_path}")
        print(f"Frames 'reconocido':                {frames_reconocido}")
        print(f"Frames 'no reconocido':             {frames_no_reconocido}")
        print(f"Accuracy (solo frames con cara):    {acc:.1f}%")

        # Estadísticas de distancias
        if len(distances_all) > 0:
            distances_np = np.array(distances_all, dtype=np.float32)
            mean_dist = float(distances_np.mean())
            std_dist = float(distances_np.std())

            print("\n--- Estadísticas de distancias descriptor ---")
            print(f"N distancias:                        {len(distances_all)}")
            print(f"Media de distancias:                 {mean_dist:.4f}")
            print(f"Desviacion estandar de distancias:   {std_dist:.4f}")
        else:
            print("\n--- Estadísticas de distancias descriptor ---")
            print("No se capturaron distancias (no hubo frames con ref y cara).")

        # Robustez temporal
        face_time = time_recognized + time_not_recognized
        print("\n--- Tiempo de reconocimiento (robustez dinámica) ---")
        print(f"Tiempo con rostro RECONOCIDO:       {time_recognized:.2f} s")
        print(f"Tiempo con rostro NO reconocido:    {time_not_recognized:.2f} s")
        if face_time > 0:
            print(f"% del tiempo con cara reconocida:   {time_recognized / face_time * 100:.1f}%")
            print(f"% del tiempo con cara no reconocida:{time_not_recognized / face_time * 100:.1f}%")
        else:
            print("No hubo tiempo con cara presente y referencia para robustez temporal.")
    else:
        print("\n(No se capturó referencia, no hay métricas de reconocimiento/tiempo).")

    print("\n--- Ángulo del rostro (roll) ---")
    print(f"Frames con ángulo estimado:         {frames_con_angulo}")
    print(f"Ángulo medio (signed):              {mean_angle:+.2f} deg")
    print(f"Ángulo medio absoluto:              {mean_abs_angle:.2f} deg")
    print(f"Ángulo máximo absoluto:             {max_abs_angle_deg:.2f} deg")

    print("\n--- Peso computacional (psutil) ---")
    print(f"CPU promedio del proceso:           {cpu_avg:.1f}%")
    print(f"CPU pico del proceso:               {max_cpu:.1f}%")
    print(f"RAM promedio del proceso:           {mem_avg:.1f} MB")
    print(f"RAM pico del proceso:               {max_mem:.1f} MB")
    print("===========================================\n")


if __name__ == "__main__":
    main()
