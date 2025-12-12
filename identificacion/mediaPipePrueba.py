import cv2
import time
import psutil
import numpy as np
import mediapipe as mp


CAM_INDEX = 0
SHOW_METRICS = True
WINDOW_NAME = "MediaPipe FaceMesh - Experimento en tiempo real"

mp_face_mesh = mp.solutions.face_mesh

# Umbral para decidir si es la misma persona
RECOG_DIST_THRESHOLD = 0.15


def calcular_bbox_desde_landmarks(face_landmarks, frame_width, frame_height):
    """
    Calcula un bounding box (x, y, w, h) a partir de los 468 puntos de FaceMesh.
    """
    xs = [lm.x * frame_width for lm in face_landmarks.landmark]
    ys = [lm.y * frame_height for lm in face_landmarks.landmark]

    x_min = max(int(min(xs)), 0)
    y_min = max(int(min(ys)), 0)
    x_max = min(int(max(xs)), frame_width - 1)
    y_max = min(int(max(ys)), frame_height - 1)

    w = max(x_max - x_min, 1)
    h = max(y_max - y_min, 1)

    return x_min, y_min, w, h


def seleccionar_rostro_principal_facemesh(faces_info, frame_width, frame_height):
    """
    faces_info: lista de diccionarios con:
        - 'box': (x, y, w, h)
        - 'landmarks': face_landmarks
    Selecciona el rostro más grande y más centrado.
    """
    if not faces_info:
        return None

    cx_frame, cy_frame = frame_width / 2, frame_height / 2
    best_item = None
    best_score = -1e9

    for item in faces_info:
        x, y, w, h = item["box"]
        area = w * h
        cx = x + w / 2
        cy = y + h / 2
        dist_center = ((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5

        score = area - 0.1 * dist_center
        if score > best_score:
            best_score = score
            best_item = item

    return best_item


def calcular_angulo_roll_facemesh(face_landmarks, frame_width, frame_height):
    """
    Calcula el ángulo de inclinación (roll) usando dos puntos de los ojos.
    Índices típicos en FaceMesh:
      - 33: ojo izquierdo (outer)
      - 263: ojo derecho (outer)
    """
    try:
        lm_left = face_landmarks.landmark[33]
        lm_right = face_landmarks.landmark[263]
    except IndexError:
        return None, None, None, None

    lx = lm_left.x * frame_width
    ly = lm_left.y * frame_height
    rx = lm_right.x * frame_width
    ry = lm_right.y * frame_height

    dx = rx - lx
    dy = ry - ly

    if dx == 0 and dy == 0:
        return None, None, None, None

    angle_rad = np.arctan2(dy, dx)
    angle_deg = angle_rad * 180.0 / np.pi

    return angle_deg, int(lx), int(ly), int(rx), int(ry)


def compute_facemesh_descriptor(face_landmarks):
    """
    Descriptor geométrico algo más robusto:
      - Usa todos los puntos (x,y).
      - Origen: punto medio entre los ojos (33 y 263).
      - Escala: distancia entre los ojos.
      - Rotación: alinea ojos horizontalmente.
      - Devuelve vector normalizado.
    """
    # Extraemos todos los puntos en 2D
    xs = np.array([lm.x for lm in face_landmarks.landmark], dtype=np.float32)
    ys = np.array([lm.y for lm in face_landmarks.landmark], dtype=np.float32)
    coords = np.stack([xs, ys], axis=1)  # (468, 2)

    # Índices de ojos en FaceMesh
    idx_left = 33
    idx_right = 263

    # Puntos de ojos
    left = coords[idx_left]   # [x, y]
    right = coords[idx_right]

    # Punto medio entre ojos -> origen
    mid_eyes = (left + right) / 2.0
    coords -= mid_eyes  # trasladamos todo

    # Distancia entre ojos -> escala
    eye_dist = np.linalg.norm(right - left) + 1e-6
    coords /= eye_dist

    # Rotación: alineamos línea de ojos con el eje X
    dx, dy = (right - left)
    angle = np.arctan2(dy, dx)  # ángulo actual
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)
    coords = coords @ R.T  # rotamos todos los puntos

    # Normalización final suave (opcional)
    coords -= coords.mean(axis=0, keepdims=True)

    # A vector 1D
    desc = coords.flatten().astype(np.float32)
    # Normalizamos longitud del descriptor para estabilidad numérica
    norm = np.linalg.norm(desc) + 1e-6
    desc /= norm

    return desc


def main():

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # MediaPipe FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Proceso para métricas de CPU/Memoria
    process = psutil.Process()

    frame_count = 0

    # tiempos
    total_proc_time = 0.0
    total_det_time = 0.0

    # deteccion
    frames_con_rostro = 0
    frames_sin_rostro = 0
    frames_multiples_rostros = 0

    # reconocimiento basado en descriptor
    frames_reconocido = 0
    frames_no_reconocido = 0

    # distancias para estadística (media y desviación)
    distances_all = []

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

    # Tiempos de robustez dinámica (igual que en ORB)
    time_recognized = 0.0       # tiempo con rostro presente y reconocido
    time_not_recognized = 0.0   # tiempo con rostro presente y NO reconocido

    # Descriptor
    ref_desc = None
    ref_name = ""          # nombre de la persona de referencia
    ref_img_path = None    # ruta donde se guarda la imagen de referencia
    have_ref = False

    # Variables del frame actual
    current_desc = None
    current_face_bgr = None

    t_global_start = time.perf_counter()

    print("REGISTRO: coloca tu rostro frontalmente y pulsa 'c' para capturar referencia.")
    print("[Pulsa 'q' para salir del experimento.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la cámara.")
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]
        current_desc = None
        current_face_bgr = None

        t_frame_start = time.perf_counter()

        # Facemesh
        t_proc_start = time.perf_counter()

        # MediaPipe trabaja en RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t_det_start = time.perf_counter()
        results = face_mesh.process(rgb)
        t_det_end = time.perf_counter()

        det_time = t_det_end - t_det_start
        total_det_time += det_time

        t_proc_end = time.perf_counter()
        total_proc_time += (t_proc_end - t_proc_start)

        faces_info = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x, y, w, h = calcular_bbox_desde_landmarks(face_landmarks, frame_w, frame_h)
                faces_info.append({
                    "box": (x, y, w, h),
                    "landmarks": face_landmarks
                })

        # Actualizar métricas de detección
        num_faces = len(faces_info)
        if num_faces == 0:
            frames_sin_rostro += 1
        else:
            frames_con_rostro += 1
            if num_faces > 1:
                frames_multiples_rostros += 1

        # Seleccionar rostro principal
        main_face_info = seleccionar_rostro_principal_facemesh(faces_info, frame_w, frame_h)

        recognized = False
        dist = None
        score = 0.0

        if main_face_info is not None:
            x, y, w, h = main_face_info["box"]
            landmarks = main_face_info["landmarks"]

            # Bounding box
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0),
                          2)

            # recorte BGR de la cara actual (para guardar si se pulsa 'c')
            current_face_bgr = frame[y:y + h, x:x + w].copy()

            # Dibujar los 468 puntos
            for lm in landmarks.landmark:
                px = int(lm.x * frame_w)
                py = int(lm.y * frame_h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            # Ángulo de inclinación (roll)
            angle_deg, lx, ly, rx, ry = calcular_angulo_roll_facemesh(
                landmarks, frame_w, frame_h
            )

            if angle_deg is not None:
                frames_con_angulo += 1
                sum_angle_deg += angle_deg
                sum_abs_angle_deg += abs(angle_deg)
                max_abs_angle_deg = max(max_abs_angle_deg, abs(angle_deg))

                # Dibujar ojos y línea
                cv2.circle(frame, (lx, ly), 3, (255, 0, 0), -1)
                cv2.circle(frame, (rx, ry), 3, (255, 0, 0), -1)
                cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 0), 2)

                text_angle = f"Angulo (roll): {angle_deg:+5.1f} deg"
                cv2.putText(frame, text_angle,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            # Descriptor FaceMesh y reconocimiento
            desc_cur = compute_facemesh_descriptor(landmarks)
            current_desc = desc_cur

            if have_ref and ref_desc is not None:
                # Distancia Euclídea entre descriptores
                dist = float(np.linalg.norm(desc_cur - ref_desc))

                # Guardar distancia para estadísticas
                distances_all.append(dist)

                # Pseudo-similaridad
                score = 1.0 / (1.0 + dist)

                if dist <= RECOG_DIST_THRESHOLD:
                    recognized = True
                    frames_reconocido += 1
                else:
                    recognized = False
                    frames_no_reconocido += 1

                color = (0, 255, 0) if recognized else (0, 0, 255)
                text_score = f"dist: {dist:.3f}  sim: {score:.3f}"

                # Etiqueta con nombre
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

        # Metricas
        t_frame_end = time.perf_counter()
        frame_time = t_frame_end - t_frame_start
        fps_instant = 1.0 / frame_time if frame_time > 0 else 0.0

        # Tiempo de robustez dinamica
        if have_ref and main_face_info is not None:
            if recognized:
                time_recognized += frame_time
            else:
                time_not_recognized += frame_time

        if SHOW_METRICS:
            avg_det_time_ms = (total_det_time / frame_count) * 1000.0
            avg_proc_time_ms = (total_proc_time / frame_count) * 1000.0
            fps_global = frame_count / (t_frame_end - t_global_start)

            mode_str = "RECONOCIMIENTO" if have_ref else "REGISTRO"
            cv2.putText(frame, f"Modo: {mode_str}",
                        (10, 20),
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

            # CPU y memoria del proceso
            cpu_percent = process.cpu_percent(interval=0)
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

            # Mostrar nombre de referencia, si existe
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


        cv2.imshow(WINDOW_NAME, frame)

        # Teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Captura de descriptor e imagen de referencia
        if not have_ref and key == ord('c') and current_desc is not None:
            ref_desc = current_desc.copy()

            # pedir el nombre por consola
            try:
                nombre = input("Nombre de la persona de referencia: ").strip()
            except EOFError:
                nombre = ""

            if nombre == "":
                nombre = "Persona"

            ref_name = nombre

            # guardar la imagen del rostro de referencia, si la tenemos
            if current_face_bgr is not None:
                safe_name = ref_name.replace(" ", "_")
                ref_img_path = f"ref_{safe_name}.png"
                cv2.imwrite(ref_img_path, current_face_bgr)
                print(f"[INFO] Imagen de referencia guardada en: {ref_img_path}")

            have_ref = True
            print(f"[INFO] Descriptor de referencia capturado para: {ref_name}")


    t_global_end = time.perf_counter()
    total_time = t_global_end - t_global_start

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

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

    print("\n===== RESULTADOS DEL EXPERIMENTO MEDIAPIPE FACEMESH (DESCRIPTOR) =====")
    print(f"Frames totales procesados:              {frame_count}")
    print(f"Tiempo total:                           {total_time:.2f} s")
    print(f"FPS global promedio:                    {fps_global:.2f}")
    print(f"Tiempo detección promedio:              {avg_det_time_ms:.2f} ms")
    print(f"Tiempo proc (RGB+FaceMesh) promedio:    {avg_proc_time_ms:.2f} ms")

    print("\n--- Detección de rostros (aprox) ---")
    print(f"Frames con al menos 1 rostro:           {frames_con_rostro} "
          f"({frames_con_rostro / frame_count * 100:.1f}%)")
    print(f"Frames sin rostro:                      {frames_sin_rostro} "
          f"({frames_sin_rostro / frame_count * 100:.1f}%)")
    print(f"Frames con múltiples rostros:           {frames_multiples_rostros} "
          f"({frames_multiples_rostros / frame_count * 100:.1f}%)")

    if have_ref:
        total_eval = frames_reconocido + frames_no_reconocido
        acc = (frames_reconocido / total_eval * 100.0) if total_eval > 0 else 0.0
        print("\n--- Reconocimiento (descriptor FaceMesh) ---")
        print(f"Persona de referencia:                  {ref_name}")
        if ref_img_path:
            print(f"Imagen de referencia:                   {ref_img_path}")
        print(f"Frames 'reconocido':                    {frames_reconocido}")
        print(f"Frames 'no reconocido':                 {frames_no_reconocido}")
        print(f"Accuracy (solo frames con cara):        {acc:.1f}%")

    # --- Estadísticas de distancias ---
    if len(distances_all) > 0:
        distances_np = np.array(distances_all, dtype=np.float32)
        mean_dist = float(distances_np.mean())
        std_dist = float(distances_np.std())

        print("\n--- Estadísticas de distancias descriptor ---")
        print(f"N distancias:                            {len(distances_all)}")
        print(f"Media de distancias:                     {mean_dist:.4f}")
        print(f"Desviacion estandar de distancias:       {std_dist:.4f}")
    else:
        print("\n--- Estadísticas de distancias descriptor ---")
        print("No se capturaron distancias (no hubo frames con ref y cara).")

    # Tiempo
    face_time = time_recognized + time_not_recognized
    print("\n--- Tiempo de reconocimiento (robustez dinámica) ---")
    print(f"Tiempo con rostro RECONOCIDO:           {time_recognized:.2f} s")
    print(f"Tiempo con rostro NO reconocido:        {time_not_recognized:.2f} s")
    if face_time > 0:
        print(f"% del tiempo con cara reconocida:       {time_recognized / face_time * 100:.1f}%")
        print(f"% del tiempo con cara no reconocida:    {time_not_recognized / face_time * 100:.1f}%")
    else:
        print("No hubo tiempo con cara presente y referencia para medir robustez temporal.")

    print("\n--- Ángulo del rostro (roll) ---")
    print(f"Frames con ángulo estimado:             {frames_con_angulo}")
    print(f"Ángulo medio (signed):                  {mean_angle:+.2f} deg")
    print(f"Ángulo medio absoluto:                  {mean_abs_angle:.2f} deg")
    print(f"Ángulo máximo absoluto:                 {max_abs_angle_deg:.2f} deg")

    print("\n--- Peso computacional (psutil) ---")
    print(f"CPU promedio del proceso:               {cpu_avg:.1f}%")
    print(f"CPU pico del proceso:                   {max_cpu:.1f}%")
    print(f"RAM promedio del proceso:               {mem_avg:.1f} MB")
    print(f"RAM pico del proceso:                   {max_mem:.1f} MB")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
