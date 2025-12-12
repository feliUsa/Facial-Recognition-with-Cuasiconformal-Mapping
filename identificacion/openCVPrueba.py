import cv2
import time
import psutil
import numpy as np


CAM_INDEX = 0
SHOW_METRICS = True
WINDOW_NAME = "ORB + YuNet - Experimento en tiempo real"

ORB_NFEATURES = 1000           # número de keypoints ORB
GOOD_MATCH_RATIO = 0.75        # ratio test de Lowe para matches
RECOG_THRESHOLD = 0.15         # umbral de "score" para decir Reconocido

FACE_SIZE = (160, 160)         # tamaño normalizado del rostro para ORB

# Modelo YuNet
YUNET_MODEL_PATH = "./modelos/face_detection_yunet_2023mar.onnx"
YUNET_SCORE_THRESH = 0.9
YUNET_NMS_THRESH = 0.3
YUNET_TOP_K = 5000


def seleccionar_rostro_principal_yunet(faces, frame_width, frame_height):
    """
    faces: array Nx15 de YuNet:
      [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
    Devuelve el índice del rostro principal (más grande y centrado).
    """
    if faces is None or len(faces) == 0:
        return None

    cx_frame, cy_frame = frame_width / 2.0, frame_height / 2.0
    best_idx = None
    best_score = -1e9

    for i, face in enumerate(faces):
        x, y, w, h = face[:4]
        area = w * h
        cx = x + w / 2.0
        cy = y + h / 2.0
        dist_center = ((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5

        score = area - 0.1 * dist_center
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def preprocess_face(gray_face):
    """
    Mejora la imagen del rostro antes de pasarla a ORB:
    - Redimensiona a tamaño fijo
    - Ecualiza el histograma (mejor contraste)
    """
    face_resized = cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_LINEAR)
    face_eq = cv2.equalizeHist(face_resized)
    return face_eq


def main():

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    yunet = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (frame_w, frame_h),
        score_threshold=YUNET_SCORE_THRESH,
        nms_threshold=YUNET_NMS_THRESH,
        top_k=YUNET_TOP_K
    )

    # ORB + Matcher
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Proceso para métricas de CPU/Memoria
    process = psutil.Process()


    frame_count = 0

    # tiempos
    total_proc_time = 0.0        # detección + ORB + matching (por frame)
    total_det_yunet = 0.0        # solo detección YuNet
    total_orb_desc = 0.0         # solo detectAndCompute
    total_match_time = 0.0       # solo matching

    # detección / reconocimiento
    frames_con_rostro = 0
    frames_sin_rostro = 0
    frames_reconocido = 0
    frames_no_reconocido = 0

    # ORB específico
    frames_with_orb = 0          # frames donde se ejecuto ORB y hubo kp_cur
    total_kp_cur = 0
    total_good_matches = 0

    # Estadísticas de "score" (good_matches / keypoints_ref)
    scores_all = []

    # Angulo (roll) de la cara
    frames_con_angulo = 0
    sum_angle_deg = 0.0
    sum_abs_angle_deg = 0.0
    max_abs_angle_deg = 0.0

    # CPU / MEM
    sum_cpu = 0.0
    sum_mem = 0.0
    max_cpu = 0.0
    max_mem = 0.0

    # Tiempos de robustez dinámica
    time_recognized = 0.0       # tiempo total con cara presente y reconocida
    time_not_recognized = 0.0   # tiempo total con cara presente y NO reconocida

    t_global_start = time.perf_counter()

    # Rostro de referencia (se captura en modo REGISTRO)
    ref_kp = None
    ref_des = None
    have_ref = False
    ref_name = ""           # nombre de la persona de referencia
    ref_img_path = None     # ruta de la imagen de referencia

    current_face_gray = None   # ROI en gris
    current_face_bgr = None    # ROI en BGR para guardar imagen

    print("[INFO] Modo REGISTRO: coloca tu rostro frontalmente y pulsa 'c' para capturar referencia.")
    print("[INFO] Pulsa 'q' para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No se pudo leer un frame de la cámara.")
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        t_frame_start = time.perf_counter()

        # Escala de grises para ORB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Procesamiento Yunet + ROI
        t_proc_start = time.perf_counter()

        # Actualizar tamaño de entrada de YuNet
        yunet.setInputSize((frame_w, frame_h))

        # Detección de rostro con YuNet
        t_det_start = time.perf_counter()
        _, faces = yunet.detect(frame)  # frame en BGR
        t_det_end = time.perf_counter()
        total_det_yunet += (t_det_end - t_det_start)

        # Actualizar métricas de detección
        if faces is None or len(faces) == 0:
            main_face = None
            frames_sin_rostro += 1
        else:
            frames_con_rostro += 1
            best_idx = seleccionar_rostro_principal_yunet(faces, frame_w, frame_h)
            main_face = faces[best_idx] if best_idx is not None else None

        recognized = False
        score = 0.0
        current_face_gray = None
        current_face_bgr = None

        if main_face is not None:
            # YuNet devuelve floats
            x, y, w, h = main_face[:4]
            x = int(max(0, x))
            y = int(max(0, y))
            w = int(max(1, w))
            h = int(max(1, h))

            # dibujar bounding box
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0),
                          2)

            # ROI del rostro (en gris y en BGR)
            current_face_gray = gray[y:y + h, x:x + w]
            current_face_bgr = frame[y:y + h, x:x + w].copy()

            # Angulo del rostro
            # Formato YuNet:
            # x, y, w, h,
            # x_re, y_re, x_le, y_le,
            # x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score
            x_re, y_re, x_le, y_le = main_face[4:8]

            angle_deg = None
            if x_re > 0 and x_le > 0:
                dx = x_re - x_le
                dy = y_re - y_le
                angle_rad = np.arctan2(dy, dx)
                angle_deg = angle_rad * 180.0 / np.pi

                frames_con_angulo += 1
                sum_angle_deg += angle_deg
                sum_abs_angle_deg += abs(angle_deg)
                max_abs_angle_deg = max(max_abs_angle_deg, abs(angle_deg))

                # Dibujar ojos
                cv2.circle(frame, (int(x_re), int(y_re)), 3, (255, 0, 0), -1)
                cv2.circle(frame, (int(x_le), int(y_le)), 3, (255, 0, 0), -1)
                cv2.line(frame, (int(x_le), int(y_le)), (int(x_re), int(y_re)), (255, 0, 0), 2)

                text_angle = f"Angulo (roll): {angle_deg:+5.1f} deg"
                cv2.putText(frame, text_angle,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            if current_face_gray is not None and current_face_gray.size > 0:
                # Preprocesar rostro para ORB (normalizar tamaño + contraste)
                face_for_orb = preprocess_face(current_face_gray)

                # Keypoints + descriptores del frame actual
                t_orb_start = time.perf_counter()
                kp_cur, des_cur = orb.detectAndCompute(face_for_orb, None)
                t_orb_end = time.perf_counter()
                total_orb_desc += (t_orb_end - t_orb_start)

                # Keypoints ORB en el frame
                if kp_cur is not None and len(kp_cur) > 0:
                    frames_with_orb += 1
                    total_kp_cur += len(kp_cur)

                    # relación de escala entre la cara normalizada y el bbox real
                    scale_x = w / float(FACE_SIZE[0])
                    scale_y = h / float(FACE_SIZE[1])

                    for kp in kp_cur:
                        u, v = kp.pt  # coordenadas en la imagen 160x160
                        px = int(x + u * scale_x)
                        py = int(y + v * scale_y)
                        cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)


                # Si ya tenemos rostro de referencia, hacemos matching
                if have_ref and ref_des is not None and des_cur is not None and kp_cur is not None and len(kp_cur) > 0:
                    t_match_start = time.perf_counter()
                    matches = bf.knnMatch(ref_des, des_cur, k=2)

                    good_matches = []
                    for m_n in matches:
                        if len(m_n) < 2:
                            continue
                        m, n = m_n
                        if m.distance < GOOD_MATCH_RATIO * n.distance:
                            good_matches.append(m)
                    t_match_end = time.perf_counter()
                    total_match_time += (t_match_end - t_match_start)

                    total_good_matches += len(good_matches)

                    if len(ref_kp) > 0:
                        score = len(good_matches) / float(len(ref_kp))
                    else:
                        score = 0.0

                    # Guardar score para estadísticas
                    scores_all.append(score)

                    if score >= RECOG_THRESHOLD:
                        recognized = True
                        frames_reconocido += 1
                    else:
                        frames_no_reconocido += 1

                    # Mostrar score y etiqueta
                    color = (0, 255, 0) if recognized else (0, 0, 255)
                    text_score = f"Score ORB: {score:.3f}"

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
                else:
                    # Si ya había referencia pero no se pudieron calcular descriptores,
                    # lo contamos como "no reconocido".
                    if have_ref:
                        frames_no_reconocido += 1
        else:
            frames_sin_rostro += 1

        t_proc_end = time.perf_counter()
        proc_time = t_proc_end - t_proc_start
        total_proc_time += proc_time


        t_frame_end = time.perf_counter()
        frame_time = t_frame_end - t_frame_start
        fps_instant = 1.0 / frame_time if frame_time > 0 else 0.0

        # tiempos robustez dinámica
        if have_ref and main_face is not None:
            if recognized:
                time_recognized += frame_time
            else:
                time_not_recognized += frame_time

        if SHOW_METRICS:
            fps_global = frame_count / (t_frame_end - t_global_start)
            avg_proc_time_ms = (total_proc_time / frame_count) * 1000.0

            mode_str = "RECONOCIMIENTO" if have_ref else "REGISTRO"
            cv2.putText(frame, f"Modo: {mode_str}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            text1 = f"FPS inst: {fps_instant:5.1f}"
            text2 = f"FPS global: {fps_global:5.1f}"
            text3 = f"T_proc_prom: {avg_proc_time_ms:6.2f} ms"

            cv2.putText(frame, text1, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text2, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text3, (10, 95),
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

            cv2.putText(frame, text4, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, text5, (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Mostrar referencia si existe
            if have_ref and ref_name:
                cv2.putText(frame,
                            f"Ref: {ref_name}",
                            (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            # Instrucción de registro
            if not have_ref:
                cv2.putText(frame,
                            "Pulsa 'c' para capturar rostro de referencia",
                            (10, frame_h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        cv2.imshow(WINDOW_NAME, frame)

        # Teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Captura de rostro de referencia
        if not have_ref and key == ord('c') and current_face_gray is not None:
            face_for_orb_ref = preprocess_face(current_face_gray)
            kp_ref, des_ref = orb.detectAndCompute(face_for_orb_ref, None)
            if des_ref is None or len(kp_ref) < 10:
                print("[WARN] No se detectaron suficientes características ORB en el rostro de referencia.")
            else:
                ref_kp = kp_ref
                ref_des = des_ref

                # Pedir nombre por consola
                try:
                    nombre = input("Nombre de la persona de referencia: ").strip()
                except EOFError:
                    nombre = ""

                if nombre == "":
                    nombre = "Persona"

                ref_name = nombre

                # Guardar imagen de referencia
                if current_face_bgr is not None:
                    safe_name = ref_name.replace(" ", "_")
                    ref_img_path = f"ref_orb_{safe_name}.png"
                    cv2.imwrite(ref_img_path, current_face_bgr)
                    print(f"[INFO] Imagen de referencia guardada en: {ref_img_path}")

                have_ref = True
                print(f"[INFO] Rostro de referencia capturado para {ref_name} con {len(ref_kp)} keypoints ORB.")

    t_global_end = time.perf_counter()
    total_time = t_global_end - t_global_start

    cap.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print("[INFO] No se procesaron frames.")
        return

    avg_proc_time_ms = (total_proc_time / frame_count) * 1000.0
    fps_global = frame_count / total_time

    avg_det_yunet_ms = (total_det_yunet / frame_count) * 1000.0

    if frames_with_orb > 0:
        avg_orb_desc_ms = (total_orb_desc / frames_with_orb) * 1000.0
        avg_match_ms = (total_match_time / frames_with_orb) * 1000.0
        avg_kp_cur = total_kp_cur / frames_with_orb
        avg_good_matches = total_good_matches / frames_with_orb
    else:
        avg_orb_desc_ms = 0.0
        avg_match_ms = 0.0
        avg_kp_cur = 0.0
        avg_good_matches = 0.0

    cpu_avg = sum_cpu / frame_count if frame_count > 0 else 0.0
    mem_avg = sum_mem / frame_count if frame_count > 0 else 0.0

    if frames_con_angulo > 0:
        mean_angle = sum_angle_deg / frames_con_angulo
        mean_abs_angle = sum_abs_angle_deg / frames_con_angulo
    else:
        mean_angle = 0.0
        mean_abs_angle = 0.0

    print("\n===== RESULTADOS DEL EXPERIMENTO ORB + YuNet (OpenCV) =====")
    print(f"Frames totales procesados:              {frame_count}")
    print(f"Tiempo total:                           {total_time:.2f} s")
    print(f"FPS global promedio:                    {fps_global:.2f}")
    print(f"Tiempo proc. ORB+YuNet promedio:        {avg_proc_time_ms:.2f} ms")
    print(f"  - Tiempo promedio deteccion YuNet:    {avg_det_yunet_ms:.2f} ms")
    print(f"  - Tiempo promedio ORB detect+desc:    {avg_orb_desc_ms:.2f} ms")
    print(f"  - Tiempo promedio matching:           {avg_match_ms:.2f} ms")

    print("\n--- Detección / reconocimiento ---")
    print(f"Frames con al menos 1 rostro:           {frames_con_rostro} "
          f"({frames_con_rostro / frame_count * 100:.1f}%)")
    print(f"Frames sin rostro:                      {frames_sin_rostro} "
          f"({frames_sin_rostro / frame_count * 100:.1f}%)")
    if have_ref:
        total_eval = frames_reconocido + frames_no_reconocido
        acc = (frames_reconocido / total_eval * 100.0) if total_eval > 0 else 0.0
        print(f"Persona de referencia:                  {ref_name}")
        if ref_img_path:
            print(f"Imagen de referencia:                   {ref_img_path}")
        print(f"Frames 'reconocido':                    {frames_reconocido}")
        print(f"Frames 'no reconocido':                 {frames_no_reconocido}")
        print(f"Accuracy (sobre frames con referencia): {acc:.1f}%")

    print("\n--- ORB (keypoints / matches) ---")
    print(f"Frames con ORB ejecutado:               {frames_with_orb}")
    print(f"Keypoints promedio por frame (cara):    {avg_kp_cur:.1f}")
    print(f"Good matches promedio por frame:        {avg_good_matches:.1f}")

    # Estadísticas de score ORB (similaridad)
    if len(scores_all) > 0:
        scores_np = np.array(scores_all, dtype=np.float32)
        mean_score = float(scores_np.mean())
        std_score = float(scores_np.std())

        print("\n--- Estadísticas del score ORB ---")
        print(f"N scores:                              {len(scores_all)}")
        print(f"Media del score:                       {mean_score:.4f}")
        print(f"Desviacion estandar del score:         {std_score:.4f}")
    else:
        print("\n--- Estadísticas del score ORB ---")
        print("No se registraron scores (no hubo matching con referencia).")

    # Tiempos de reconocimiento dinámico
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
    print("===============================================\n")


if __name__ == "__main__":
    main()
