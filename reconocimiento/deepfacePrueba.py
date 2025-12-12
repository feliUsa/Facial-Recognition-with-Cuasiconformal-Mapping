import cv2
import time
import psutil
import numpy as np
import mediapipe as mp
from deepface import DeepFace


CAM_INDEX = 0
SHOW_METRICS = True

# "ArcFace" / "SFace"
MODEL_NAME = "SFace"
WINDOW_NAME = f"Mediapipe FaceMesh + DeepFace ({MODEL_NAME}) - Experimento en tiempo real"

# Umbral de similitud coseno para decir "Reconocido"
SIM_THRESHOLD = 0.6

mp_face_mesh = mp.solutions.face_mesh


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
        - 'roi_bgr': recorte BGR
    Selecciona el rostro más grande y más centrado.
    """
    if not faces_info:
        return None

    cx_frame, cy_frame = frame_width / 2.0, frame_height / 2.0
    best_item = None
    best_score = -1e9

    for item in faces_info:
        x, y, w, h = item["box"]
        area = w * h
        cx = x + w / 2.0
        cy = y + h / 2.0
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


def compute_cosine_similarity(vec1, vec2):
    v1 = vec1.astype(np.float32)
    v2 = vec2.astype(np.float32)
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    return float(np.dot(v1, v2) / (n1 * n2))


def main():

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Inicializar MediaPipe FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # precargar el modelo DeepFace para que el primer embedding
    # no sea tan lento. NO se pasa este objeto a represent().
    print(f"[INFO] Cargando modelo DeepFace: {MODEL_NAME}...")
    _ = DeepFace.build_model(MODEL_NAME)
    print("[INFO] Modelo cargado.")

    process = psutil.Process()


    frame_count = 0

    total_proc_time = 0.0     # FaceMesh + DeepFace por frame
    total_det_time = 0.0      # SOLO FaceMesh.process
    total_rep_time = 0.0      # SOLO DeepFace.represent
    frames_with_rep = 0       # cuántos frames calcularon embedding

    frames_con_rostro = 0
    frames_sin_rostro = 0
    frames_multiples_rostros = 0

    have_ref = False
    ref_embedding = None
    ref_name = ""
    ref_img_path = None

    frames_reconocido = 0
    frames_no_reconocido = 0

    time_recognized = 0.0
    time_not_recognized = 0.0

    frames_con_angulo = 0
    sum_angle_deg = 0.0
    sum_abs_angle_deg = 0.0
    max_abs_angle_deg = 0.0

    distances_all = []
    sims_all = []

    sum_cpu = 0.0
    sum_mem = 0.0
    max_cpu = 0.0
    max_mem = 0.0

    t_global_start = time.perf_counter()

    print(f"[INFO] FaceMesh + DeepFace ({MODEL_NAME})")
    print("[INFO] Modo REGISTRO: coloca tu rostro y pulsa 'c' para capturar referencia.")
    print("[INFO] Pulsa 'q' para salir.\n")

    current_embedding = None
    current_face_bgr = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No se pudo leer un frame de la cámara.")
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        t_frame_start = time.perf_counter()
        t_proc_start = time.perf_counter()

        # Facemesh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t_det_start = time.perf_counter()
        results = face_mesh.process(rgb)
        t_det_end = time.perf_counter()
        det_time = t_det_end - t_det_start
        total_det_time += det_time

        faces_info = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x, y, w, h = calcular_bbox_desde_landmarks(
                    face_landmarks, frame_w, frame_h
                )
                roi_bgr = frame[y:y + h, x:x + w].copy()
                faces_info.append({
                    "box": (x, y, w, h),
                    "landmarks": face_landmarks,
                    "roi_bgr": roi_bgr
                })

        # Métricas de detección
        num_faces = len(faces_info)
        if num_faces == 0:
            frames_sin_rostro += 1
        else:
            frames_con_rostro += 1
            if num_faces > 1:
                frames_multiples_rostros += 1

        # Seleccionar rostro principal
        main_face_info = seleccionar_rostro_principal_facemesh(
            faces_info, frame_w, frame_h
        )

        recognized = False
        dist_l2 = None
        sim_cos = None

        current_embedding = None
        current_face_bgr = None

        if main_face_info is not None:
            x, y, w, h = main_face_info["box"]
            landmarks = main_face_info["landmarks"]
            current_face_bgr = main_face_info["roi_bgr"]

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Dibujar 468 puntos
            for lm in landmarks.landmark:
                px = int(lm.x * frame_w)
                py = int(lm.y * frame_h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            # Ángulo (roll)
            angle_deg, lx, ly, rx, ry = calcular_angulo_roll_facemesh(
                landmarks, frame_w, frame_h
            )
            if angle_deg is not None:
                frames_con_angulo += 1
                sum_angle_deg += angle_deg
                sum_abs_angle_deg += abs(angle_deg)
                max_abs_angle_deg = max(max_abs_angle_deg, abs(angle_deg))

                cv2.circle(frame, (lx, ly), 3, (255, 0, 0), -1)
                cv2.circle(frame, (rx, ry), 3, (255, 0, 0), -1)
                cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 0), 2)

                text_angle = f"Angulo (roll): {angle_deg:+5.1f} deg"
                cv2.putText(frame, text_angle,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            # DeepFace
            if current_face_bgr is not None and current_face_bgr.size > 0:
                t_rep_start = time.perf_counter()
                try:
                    reps = DeepFace.represent(
                        img_path=current_face_bgr,   # numpy array de la cara
                        model_name=MODEL_NAME,
                        detector_backend="skip",     # ya está recortada
                        enforce_detection=False
                    )
                except Exception as e:
                    print("[ERROR represent]:", e)
                    reps = []

                t_rep_end = time.perf_counter()
                total_rep_time += (t_rep_end - t_rep_start)
                frames_with_rep += 1

                # Extraer embedding
                if isinstance(reps, list) and len(reps) > 0:
                    emb = reps[0].get("embedding", None)
                elif isinstance(reps, dict):
                    emb = reps.get("embedding", None)
                else:
                    emb = None

                if emb is not None:
                    current_embedding = np.array(emb, dtype=np.float32)

                    # Comparar con la referencia
                    if have_ref and ref_embedding is not None:
                        dist_l2 = float(np.linalg.norm(current_embedding - ref_embedding))
                        sim_cos = compute_cosine_similarity(current_embedding, ref_embedding)

                        distances_all.append(dist_l2)
                        sims_all.append(sim_cos)

                        if sim_cos >= SIM_THRESHOLD:
                            recognized = True
                            frames_reconocido += 1
                        else:
                            recognized = False
                            frames_no_reconocido += 1

                        color = (0, 255, 0) if recognized else (0, 0, 255)
                        text_score = f"sim: {sim_cos:.3f}  L2: {dist_l2:.3f}"

                        if recognized:
                            label = f"Reconocido ({ref_name})" if ref_name else "Reconocido"
                        else:
                            label = f"No es ({ref_name})" if ref_name else "No reconocido"

                        cv2.putText(frame, text_score,
                                    (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2)
                        cv2.putText(frame, label,
                                    (x, y + h + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2)

        # Fin procesamiento core
        t_proc_end = time.perf_counter()
        total_proc_time += (t_proc_end - t_proc_start)

        # Metricas
        t_frame_end = time.perf_counter()
        frame_time = t_frame_end - t_frame_start
        fps_instant = 1.0 / frame_time if frame_time > 0 else 0.0

        # Robustez temporal
        if have_ref and main_face_info is not None and current_embedding is not None:
            if recognized:
                time_recognized += frame_time
            else:
                time_not_recognized += frame_time

        if SHOW_METRICS:
            fps_global = frame_count / (t_frame_end - t_global_start)
            avg_det_ms = (total_det_time / frame_count) * 1000.0
            avg_proc_ms = (total_proc_time / frame_count) * 1000.0
            avg_rep_ms = (total_rep_time / frames_with_rep) * 1000.0 if frames_with_rep > 0 else 0.0

            mode_str = "RECONOCIMIENTO" if have_ref else "REGISTRO"
            cv2.putText(frame, f"Modelo: {MODEL_NAME}  Modo: {mode_str}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            text1 = f"FPS inst: {fps_instant:5.1f}"
            text2 = f"FPS global: {fps_global:5.1f}"
            text3 = f"T_det_prom (FaceMesh): {avg_det_ms:6.2f} ms"
            text4 = f"T_rep_prom (DeepFace): {avg_rep_ms:6.2f} ms"
            text5b = f"T_proc_prom total:      {avg_proc_ms:6.2f} ms"

            cv2.putText(frame, text1, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text2, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text3, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text4, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text5b, (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cpu_percent = process.cpu_percent(interval=0)
            mem_mb = process.memory_info().rss / (1024 * 1024)

            sum_cpu += cpu_percent
            sum_mem += mem_mb
            max_cpu = max(max_cpu, cpu_percent)
            max_mem = max(max_mem, mem_mb)

            text_cpu = f"CPU proc: {cpu_percent:5.1f}%"
            text_mem = f"MEM proc: {mem_mb:6.1f} MB"

            cv2.putText(frame, text_cpu, (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, text_mem, (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if have_ref and ref_name:
                cv2.putText(frame,
                            f"Ref: {ref_name}",
                            (10, 220),
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

        # Captura de referencia
        if not have_ref and key == ord('c') and current_embedding is not None:
            ref_embedding = current_embedding.copy()

            try:
                nombre = input("Nombre de la persona de referencia: ").strip()
            except EOFError:
                nombre = ""

            if nombre == "":
                nombre = "Persona"

            ref_name = nombre

            if current_face_bgr is not None:
                safe_name = ref_name.replace(" ", "_")
                ref_img_path = f"ref_facemesh_deepface_{MODEL_NAME}_{safe_name}.png"
                cv2.imwrite(ref_img_path, current_face_bgr)
                print(f"[INFO] Imagen de referencia guardada en: {ref_img_path}")

            have_ref = True
            print(f"[INFO] Embedding de referencia capturado para {ref_name} (modelo {MODEL_NAME}).")


    t_global_end = time.perf_counter()
    total_time = t_global_end - t_global_start

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if frame_count == 0:
        print("[INFO] No se procesaron frames.")
        return

    avg_det_ms = (total_det_time / frame_count) * 1000.0
    avg_proc_ms = (total_proc_time / frame_count) * 1000.0
    avg_rep_ms = (total_rep_time / frames_with_rep) * 1000.0 if frames_with_rep > 0 else 0.0
    fps_global = frame_count / total_time

    cpu_avg = sum_cpu / frame_count if frame_count > 0 else 0.0
    mem_avg = sum_mem / frame_count if frame_count > 0 else 0.0

    if frames_con_angulo > 0:
        mean_angle = sum_angle_deg / frames_con_angulo
        mean_abs_angle = sum_abs_angle_deg / frames_con_angulo
    else:
        mean_angle = 0.0
        mean_abs_angle = 0.0

    print(f"\n===== RESULTADOS FaceMesh + DeepFace ({MODEL_NAME}) =====")
    print(f"Frames totales procesados:              {frame_count}")
    print(f"Tiempo total:                           {total_time:.2f} s")
    print(f"FPS global promedio:                    {fps_global:.2f}")
    print(f"Tiempo det FaceMesh promedio:           {avg_det_ms:.2f} ms")
    print(f"Tiempo rep DeepFace promedio:           {avg_rep_ms:.2f} ms")
    print(f"Tiempo proc total promedio:             {avg_proc_ms:.2f} ms")

    print("\n--- Detección de rostros (FaceMesh) ---")
    print(f"Frames con al menos 1 rostro:           {frames_con_rostro} "
          f"({frames_con_rostro / frame_count * 100:.1f}%)")
    print(f"Frames sin rostro:                      {frames_sin_rostro} "
          f"({frames_sin_rostro / frame_count * 100:.1f}%)")
    print(f"Frames con múltiples rostros:           {frames_multiples_rostros} "
          f"({frames_multiples_rostros / frame_count * 100:.1f}%)")

    if have_ref:
        total_eval = frames_reconocido + frames_no_reconocido
        acc = (frames_reconocido / total_eval * 100.0) if total_eval > 0 else 0.0
        print("\n--- Reconocimiento (DeepFace) ---")
        print(f"Persona de referencia:                  {ref_name}")
        if ref_img_path:
            print(f"Imagen de referencia:                   {ref_img_path}")
        print(f"Frames 'reconocido':                    {frames_reconocido}")
        print(f"Frames 'no reconocido':                 {frames_no_reconocido}")
        print(f"Accuracy (solo frames con ref+cara):    {acc:.1f}%")

        face_time = time_recognized + time_not_recognized
        print("\n--- Tiempo de reconocimiento (robustez temporal) ---")
        print(f"Tiempo con rostro RECONOCIDO:           {time_recognized:.2f} s")
        print(f"Tiempo con rostro NO reconocido:        {time_not_recognized:.2f} s")
        if face_time > 0:
            print(f"% del tiempo con cara reconocida:       {time_recognized / face_time * 100:.1f}%")
            print(f"% del tiempo con cara no reconocida:    {time_not_recognized / face_time * 100:.1f}%")
        else:
            print("No hubo tiempo con cara presente y referencia para medir robustez temporal.")
    else:
        print("\n[INFO] No se llegó a capturar referencia; no hay métricas de reconocimiento.")

    if len(distances_all) > 0:
        d_np = np.array(distances_all, dtype=np.float32)
        s_np = np.array(sims_all, dtype=np.float32)
        print("\n--- Estadísticas de distancias / similitud ---")
        print(f"N mediciones:                            {len(distances_all)}")
        print(f"Distancia L2 media:                     {float(d_np.mean()):.4f} "
              f"(std: {float(d_np.std()):.4f})")
        print(f"Similitud coseno media:                 {float(s_np.mean()):.4f} "
              f"(std: {float(s_np.std()):.4f})")
    else:
        print("\n--- Estadísticas de distancias / similitud ---")
        print("No se capturaron distancias (no hubo frames con ref y cara).")

    print("\n--- Peso computacional (psutil) ---")
    print(f"CPU promedio del proceso:               {cpu_avg:.1f}%")
    print(f"CPU pico del proceso:                   {max_cpu:.1f}%")
    print(f"RAM promedio del proceso:               {mem_avg:.1f} MB")
    print(f"RAM pico del proceso:                   {max_mem:.1f} MB")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
