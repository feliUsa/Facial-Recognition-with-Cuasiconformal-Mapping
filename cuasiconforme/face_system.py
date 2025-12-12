import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np

from yunet_detector import YuNetFaceDetector
from facemesh_helper import FaceMeshHelper
from stable_capture import StableCaptureController
from user_capture_manager import UserCaptureManager
from canonical_template import CanonicalFaceTemplate
from qc_warper import QuasiConformalWarper
from recognitionLibs import (
    DeepFaceRecognizer,
    BuffaloLRecognizer,
    FacenetVGGFace2Recognizer,
)

CAMERA_INDEX = 0

MODEL_PATH = (
    "/home/daniel/Universidad/experimentosLibreriasii/identificacion/modelos/face_detection_yunet_2023mar.onnx"
)

OUTPUT_DIR = "capturas_yunet_facemesh"

REGISTER_SUBDIR = "register"
LOGIN_SUBDIR = "login"

TEMPLATE_PATH = "canonical_template.npz"

REQUIRED_STABLE_FRAMES = 30

# Filtros de calidad
MIN_FACE_AREA_RATIO = 0.08       # area minima del rostro
CENTER_TOLERANCE = 0.25          # tolerancia para estar cerca del centro
ASYMMETRY_THRESHOLD = 0.25       # tolerancia para ver si el rostro esta de frente

# Limite capturas
MAX_CAPTURES_PER_USER = 5

# Cambiar a otro usuario
NO_USER_FRAME_THRESHOLD = 20

# Tamaño estandar para normalizar la ROI al guardar
QC_TARGET_SIZE = 256

# Warpeo parcial hacia la plantilla (0 → casi original, 1 → fuerte QC)
ALPHA_QC = 0.6

# Reconocimiento
USE_RECOGNITION = True

USE_DEEPFACE = True
DEEPFACE_MODEL_NAME = "ArcFace"  # "ArcFace" o "SFace"

USE_BUFFALO_L = False
USE_FACENET_VGG = False


def refine_bbox_with_landmarks(
    bbox: Optional[Tuple[int, int, int, int]],
    landmarks_img: np.ndarray,
    img_shape,
    margin_ratio: float = 0.2,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Recalcula una bbox más precisa en base a landmarks (en coordenadas de imagen)
    y aplica un margen. Retorna coordenadas enteras (x1, y1, x2, y2) ajustadas
    a la imagen, forzando una ROI cuadrada.
    """
    if landmarks_img is None or landmarks_img.size == 0:
        return bbox

    h, w, _ = img_shape

    x_min = float(np.min(landmarks_img[:, 0]))
    y_min = float(np.min(landmarks_img[:, 1]))
    x_max = float(np.max(landmarks_img[:, 0]))
    y_max = float(np.max(landmarks_img[:, 1]))

    # Añadir margen
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return bbox

    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    x1 = max(0, int(x_min - margin_x))
    y1 = max(0, int(y_min - margin_y))
    x2 = min(w - 1, int(x_max + margin_x))
    y2 = min(h - 1, int(y_max + margin_y))

    # Forzar ROI cuadrada centrada
    box_w, box_h = x2 - x1, y2 - y1
    side = max(box_w, box_h)
    cx = x1 + box_w // 2
    cy = y1 + box_h // 2

    x1 = max(0, int(cx - side // 2))
    y1 = max(0, int(cy - side // 2))
    x2 = min(w - 1, x1 + side)
    y2 = min(h - 1, y1 + side)

    return (x1, y1, x2, y2)


def check_size_and_center(
    x1, y1, x2, y2,
    frame_w, frame_h,
    min_area_ratio: float = MIN_FACE_AREA_RATIO,
    center_tolerance: float = CENTER_TOLERANCE,
) -> Tuple[bool, bool]:
    """
    Evalua si la cara:
    - Tiene un tamaño minimo (no está demasiado lejos).
    - Esta razonablemente centrada en la imagen.
    """
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    if box_w <= 0 or box_h <= 0:
        return False, False

    area_box = box_w * box_h
    area_frame = frame_w * frame_h
    if area_frame <= 0:
        return False, False

    ratio = area_box / float(area_frame)
    size_ok = ratio >= min_area_ratio

    cx_face = x1 + box_w / 2.0
    cy_face = y1 + box_h / 2.0
    cx_frame = frame_w / 2.0
    cy_frame = frame_h / 2.0

    dx_norm = abs(cx_face - cx_frame) / float(frame_w)
    dy_norm = abs(cy_face - cy_frame) / float(frame_h)

    center_ok = (dx_norm <= center_tolerance) and (dy_norm <= center_tolerance)
    return size_ok, center_ok


def is_image_sharp(img, threshold: float = 100.0) -> bool:
    """Verifica si una imagen tiene suficiente nitidez usando varianza del Laplaciano."""
    if img is None or img.size == 0:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = cv2.Laplacian(gray, cv2.CV_64F).var()
    return val > threshold


def normalize_illumination(img):
    """Normaliza la iluminacion de la imagen suavemente (gamma correction)."""
    if img is None or img.size == 0:
        return img
    img_float = img.astype(np.float32) / 255.0
    gamma = 1.1
    img_corr = np.power(img_float, 1.0 / gamma)
    return np.clip(img_corr * 255, 0, 255).astype(np.uint8)



class FaceSystem:
    """
    Orquesta todo el sistema de captura + mapeo cuasiconformal + reconocimiento.

    - register_user(nombre)  -> enrolamiento (guarda imágenes y warps)
    - login()                -> captura + QC + identificación 1:N (solo local, sin BD)
    """

    def __init__(self) -> None:
        # Directorios base
        self.output_dir = OUTPUT_DIR
        self.register_root = os.path.join(self.output_dir, REGISTER_SUBDIR)
        self.login_root = os.path.join(self.output_dir, LOGIN_SUBDIR)
        Path(self.register_root).mkdir(parents=True, exist_ok=True)
        Path(self.login_root).mkdir(parents=True, exist_ok=True)

        # Detector YuNet + FaceMesh helper
        self.detector = YuNetFaceDetector(MODEL_PATH)
        self.fm_helper = FaceMeshHelper(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )

        # Plantilla canónica + warper QC
        self.qc_template: Optional[CanonicalFaceTemplate] = None
        self.qc_warper: Optional[QuasiConformalWarper] = None

        if os.path.exists(TEMPLATE_PATH):
            print(f"Cargando plantilla canonica desde {TEMPLATE_PATH}")
            try:
                self.qc_template = CanonicalFaceTemplate.load(TEMPLATE_PATH)
                self.qc_warper = QuasiConformalWarper(self.qc_template, alpha=ALPHA_QC)
            except Exception as e:
                print(f"[QC] Error cargando plantilla: {e}")
                self.qc_template = None
                self.qc_warper = None
        else:
            print("No existe canonical_template.npz. El warpeo QC seguira vacio.")

        # Reconocimiento
        self.recognizers: Dict[str, object] = {}    # modelo → reconocedor
        self.galleries: Dict[str, Dict[str, np.ndarray]] = {}  # modelo → {user → (K,D)}

        if USE_RECOGNITION:
            # DeepFace (opcional)
            if USE_DEEPFACE:
                try:
                    self.recognizers["deepface"] = DeepFaceRecognizer(
                        model_name=DEEPFACE_MODEL_NAME
                    )
                    print(f"DeepFace inicializado con modelo {DEEPFACE_MODEL_NAME}")
                except Exception as e:
                    print(f"Error inicializando DeepFace: {e}")

            # Buffalo_L (InsightFace)
            if USE_BUFFALO_L:
                try:
                    self.recognizers["buffalo_l"] = BuffaloLRecognizer()
                    print("Buffalo_L inicializado.")
                except Exception as e:
                    print(f"Error inicializando Buffalo_L: {e}")

            # Facenet VGGFace2
            if USE_FACENET_VGG:
                try:
                    self.recognizers["facenet_vgg"] = FacenetVGGFace2Recognizer()
                    print("Facenet VGGFace2 inicializado.")
                except Exception as e:
                    print(f"Error inicializando Facenet VGGFace2: {e}")

        # Construir galerías 1:N desde las carpetas de registro
        if self.recognizers:
            self._build_galleries()
        else:
            print("Ningún modelo de reconocimiento activo; login será solo captura.")

    # Reconocimiento multimodelo
    def _build_galleries(self) -> None:
        """
        Recorre register_root y construye las galerías 1:N
        para cada modelo activo, usando las imágenes QC (warpeadas).
        """
        import glob

        self.galleries = {}

        if not os.path.isdir(self.register_root):
            print("register_root no existe, galería vacia.")
            return

        users = [
            d for d in os.listdir(self.register_root)
            if os.path.isdir(os.path.join(self.register_root, d))
        ]
        if not users:
            print("No hay usuarios registrados.")
            return

        for model_name, recognizer in self.recognizers.items():
            model_gallery: Dict[str, np.ndarray] = {}
            print(f"Construyendo galería para modelo '{model_name}'...")

            for user in users:
                user_dir = os.path.join(self.register_root, user)
                # Preferimos las imagenes ya warpeadas (QC)
                pattern_qc = os.path.join(user_dir, "register_*_qc.png")
                files = sorted(glob.glob(pattern_qc))
                if not files:
                    # Fallback a las normales si no hay QC
                    pattern_clean = os.path.join(user_dir, "register_*.png")
                    files = sorted(glob.glob(pattern_clean))

                embeddings: List[np.ndarray] = []
                for fpath in files:
                    img = cv2.imread(fpath)
                    if img is None or img.size == 0:
                        continue
                    try:
                        emb = recognizer.embed(img)
                        embeddings.append(emb)
                    except Exception as e:
                        print(f"Error embed {fpath} con {model_name}: {e}")

                if embeddings:
                    model_gallery[user] = np.stack(embeddings, axis=0)  # (K,D)

            self.galleries[model_name] = model_gallery
            print(f"Galeria '{model_name}': {len(model_gallery)} usuarios.")

        if not any(self.galleries.values()):
            print("Advertencia: Galerias vacias.")

    def _identify_multi_model(self, img_bgr: np.ndarray):
        """
        Ejecuta todos los modelos activos sobre una imagen de login
        y devuelve:
            - user_final: nombre de usuario si hay consenso,
            - details: dict por modelo con {user, dist, thr, is_match}.

        Regla: SOLO aceptamos si TODOS los modelos que devuelven match
        coinciden en el MISMO usuario.
        """
        if not self.recognizers or not self.galleries:
            return None, {}

        details: Dict[str, Dict[str, object]] = {}
        votes: List[str] = []

        for model_name, recognizer in self.recognizers.items():
            gallery = self.galleries.get(model_name, {})
            if not gallery:
                print(f"[REC] Modelo '{model_name}' sin usuarios en galería.")
                continue

            try:
                emb = recognizer.embed(img_bgr)
            except Exception as e:
                print(f"[REC] Error embed login con {model_name}: {e}")
                continue

            best_user = None
            best_dist = 999.0

            # Buscar mejor usuario en esa galería
            for user, mat in gallery.items():   # mat: (K, D)
                d_min = 999.0
                for ref in mat:
                    d = recognizer.cosine_distance(emb, ref)
                    if d < d_min:
                        d_min = d
                if d_min < best_dist:
                    best_dist = d_min
                    best_user = user

            thr = getattr(recognizer, "default_threshold", 0.35)
            is_match = best_user is not None and best_dist <= thr

            details[model_name] = {
                "user": best_user,
                "dist": best_dist,
                "thr": thr,
                "is_match": is_match,
            }

            if is_match and best_user is not None:
                votes.append(best_user)

        user_final = None
        if votes:
            unique_users = set(votes)
            if len(unique_users) == 1:
                user_final = votes[0]

        return user_final, details

    # ----------------- API pública -----------------

    def register_user(self, user_name: str) -> None:
        """
        Flujo de REGISTRO:
        - Pide frames de cámara.
        - Aplica pipeline (YuNet + FaceMesh + filtros + QC).
        - Guarda muestras en capturas_yunet_facemesh/register/<user_name>/...
        """
        user_name = user_name.strip()
        if not user_name:
            print("Nombre vacio, abortando registro.")
            return

        user_dir = os.path.join(self.register_root, user_name)
        Path(user_dir).mkdir(parents=True, exist_ok=True)

        print(f"Registrando usuario: '{user_name}'")
        print(f"Directorio: {user_dir}")
        print("Mira a la cámara. Capturaremos hasta "
              f"{MAX_CAPTURES_PER_USER} muestras estables.")

        controller = StableCaptureController(required_stable_frames=REQUIRED_STABLE_FRAMES)
        user_manager = UserCaptureManager(
            max_captures_per_user=MAX_CAPTURES_PER_USER,
            distance_threshold=0.25,
        )

        self._run_stream(
            mode="register",
            save_dir=user_dir,
            controller=controller,
            user_manager=user_manager,
            stop_after_first_save=False,
        )

        # Después de registrar, refrescamos las galerías de reconocimiento
        if self.recognizers:
            print("Re-construyendo galerias tras registro...")
            self._build_galleries()

    def login(self) -> None:
        """
        Flujo de LOGIN:
        - Pide frames de camara.
        - Aplica el mismo pipeline hasta capturas estables.
        - Guarda capturas en capturas_yunet_facemesh/login/...
        - Usa Buffalo_L + Facenet (y DeepFace si lo activas) para identificar 1:N.
        """
        print("Iniciando login.")
        print("Mira a la camara. El sistema seguira activo hasta que pulses 'q'.")

        controller = StableCaptureController(required_stable_frames=REQUIRED_STABLE_FRAMES)
        user_manager = UserCaptureManager(
            max_captures_per_user=999,
            distance_threshold=0.25,
        )

        self._run_stream(
            mode="login",
            save_dir=self.login_root,
            controller=controller,
            user_manager=user_manager,
            stop_after_first_save=False,
        )

        print("Login finalizado.")

    # ----------------- Motor interno de captura -----------------

    def _run_stream(
        self,
        mode: str,
        save_dir: str,
        controller: StableCaptureController,
        user_manager: UserCaptureManager,
        stop_after_first_save: bool,
    ) -> None:
        """
        Bucle principal: camara → detección → landmarks → QC → guardado
        (+ reconocimiento en modo login).
        """
        no_user_frames = 0

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"No se pudo abrir la camara {CAMERA_INDEX}")
            return

        win_title = f"FaceSystem - {mode.upper()}"
        print("Camara abierta. Pulsa 'q' o ESC para salir manualmente.")

        saved_count = 0
        should_stop = False

        # Mensaje persistente de login
        last_login_msg: Optional[str] = None
        last_login_color = (0, 255, 0)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame vacío, continuando...")
                continue

            h, w, _ = frame.shape
            frame_display = frame.copy()

            # Yunet
            boxes, best_box = self.detector.detect(frame)

            has_any_detection = False
            bbox_coords = None
            face_roi = None
            landmarks = None

            if best_box is not None:
                has_any_detection = True
                x1, y1, x2, y2 = best_box.astype(int)
                bbox_coords = (x1, y1, x2, y2)
            else:
                # Fallback: FaceMesh bbox
                fallback_box = self.fm_helper.detect_face_fallback_bbox(frame)
                if fallback_box is not None:
                    has_any_detection = True
                    x1, y1, x2, y2 = fallback_box.astype(int)
                    bbox_coords = (x1, y1, x2, y2)

            # 2. Recorte de ROI inicial
            if has_any_detection and bbox_coords is not None:
                x1, y1, x2, y2 = bbox_coords
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_roi = frame[y1:y2, x1:x2]

            # 3. Landmarks en la ROI (FaceMesh) + refinamiento de bbox
            if face_roi is not None and face_roi.size > 0:
                lm_roi = self.fm_helper.extract_landmarks_from_roi(face_roi)

                if lm_roi is not None:
                    lm_img = lm_roi.copy()
                    lm_img[:, 0] += x1
                    lm_img[:, 1] += y1

                    bbox_coords = refine_bbox_with_landmarks(bbox_coords, lm_img, frame.shape)
                    x1, y1, x2, y2 = bbox_coords

                    face_roi = frame[y1:y2, x1:x2]

                    if face_roi is not None and face_roi.size > 0:
                        landmarks = self.fm_helper.extract_landmarks_from_roi(face_roi)
                    else:
                        landmarks = None

                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                else:
                    landmarks = None

            # 4. Chequeos de calidad
            size_ok = False
            center_ok = False
            pose_ok = False

            if bbox_coords is not None and landmarks is not None and landmarks.size > 0:
                x1, y1, x2, y2 = bbox_coords
                size_ok, center_ok = check_size_and_center(x1, y1, x2, y2, w, h)
                pose_ok = self.fm_helper.is_frontal_enough(landmarks, ASYMMETRY_THRESHOLD)

            has_detection_for_stability = (
                has_any_detection
                and (face_roi is not None and face_roi.size > 0)
                and (landmarks is not None and landmarks.size > 0)
                and size_ok
                and center_ok
                and pose_ok
            )

            # 5. Gestión de "usuario presente / ausente"
            if has_any_detection:
                no_user_frames = 0
            else:
                no_user_frames += 1
                if no_user_frames > NO_USER_FRAME_THRESHOLD:
                    if user_manager.current_signature is not None:
                        print("Usuario ausente un rato, continuando al siguiente usuario")
                    user_manager.reset_session()
                    controller.reset()
                    no_user_frames = 0

                    # limpiar mensaje de login cuando la persona se va
                    if mode == "login":
                        last_login_msg = None

            # Filtro de nitidez e iluminación sobre la ROI actual
            if face_roi is not None and face_roi.size > 0:
                if not is_image_sharp(face_roi):
                    has_detection_for_stability = False
                else:
                    face_roi = normalize_illumination(face_roi)

            # 6. Captura estable
            stable_result = controller.update(
                has_detection=has_detection_for_stability,
                face_roi=face_roi,
                landmarks=landmarks,
            )

            # Contador de estabilidad
            status_text = f"Stable: {controller.current_stable_frames}/{REQUIRED_STABLE_FRAMES}"
            cv2.putText(
                frame_display,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if controller.current_stable_frames > 0 else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            debug_flags = []
            if has_any_detection:
                debug_flags.append("det")
            if size_ok:
                debug_flags.append("size")
            if center_ok:
                debug_flags.append("center")
            if pose_ok:
                debug_flags.append("pose")
            debug_text = " | ".join(debug_flags)

            captures_text = (
                f"Caps: {user_manager.captures_for_current_user}/"
                f"{user_manager.max_captures_per_user}"
            )

            cv2.putText(
                frame_display,
                debug_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_display,
                captures_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 7. Si hay captura estable, procesar/guardar + (login) reconocimiento
            if stable_result is not None:
                face_img = stable_result["face_image"]
                lm = stable_result["landmarks"]

                # Normalizar a tamaño estándar
                h_roi, w_roi, _ = face_img.shape
                face_img_qc = cv2.resize(
                    face_img,
                    (QC_TARGET_SIZE, QC_TARGET_SIZE),
                    interpolation=cv2.INTER_AREA,
                )

                lm_qc = lm.copy().astype(np.float32)
                scale_x = QC_TARGET_SIZE / float(w_roi)
                scale_y = QC_TARGET_SIZE / float(h_roi)
                lm_qc[:, 0] *= scale_x
                lm_qc[:, 1] *= scale_y

                lm_norm = lm_qc.copy()
                lm_norm[:, 0] /= float(QC_TARGET_SIZE)
                lm_norm[:, 1] /= float(QC_TARGET_SIZE)

                # Warpeo cuasiconformal (si existe plantilla)
                warped_img = None
                if self.qc_warper is not None:
                    try:
                        # Campo de Beltrami por triángulo (para análisis / logging)
                        mu_tri = self.qc_warper.compute_beltrami(lm_qc)
                        avg_mu = np.mean(np.abs(mu_tri))
                        print(f"Distorsión media |mu| = {avg_mu:.4f}")
                        warped_img, _ = self.qc_warper.warp(face_img_qc, lm_qc)
                    except Exception as e:
                        print(f"Error en warpeo cuasiconformal: {e}")
                        warped_img = None

                # Imagen que se usará para reconocimiento
                img_for_rec = warped_img if warped_img is not None else face_img_qc

                # --- RECONOCIMIENTO MULTI-MODELO EN LOGIN ---
                if (
                    mode == "login"
                    and USE_RECOGNITION
                    and self.recognizers
                    and self.galleries
                ):
                    user_final, rec_details = self._identify_multi_model(img_for_rec)

                    # Log por modelo
                    for mname, info in rec_details.items():
                        print(
                            f"[LOGIN:{mname}] user={info['user']} "
                            f"dist={info['dist']:.4f} thr={info['thr']:.4f} "
                            f"is_match={info['is_match']}"
                        )

                    if user_final is not None:
                        msg = f"Bienvenido, {user_final}"
                        color = (0, 255, 0)
                    elif rec_details:
                        msg = "Persona NO reconocida"
                        color = (0, 0, 255)
                    else:
                        msg = "Sin galería/modelos para reconocer."
                        color = (0, 255, 255)

                    last_login_msg = msg
                    last_login_color = color
                    print(f"[LOGIN-REC] {msg}")

                # Decisión de guardado (para registro y para debug de login)
                decision = user_manager.should_save(lm_qc)

                if decision["save"]:
                    ts = time.strftime("%Y%m%d_%H%M%S")

                    # Usamos la clave correcta del dict que devuelve UserCaptureManager
                    capture_idx = decision.get(
                        "captures_for_user",
                        1  # fallback por si en algún momento no viniera la clave
                    )

                    base = os.path.join(
                        save_dir,
                        f"{mode}_{ts}_{capture_idx}",
                    )

                    # Imagen limpia (ya normalizada a QC_TARGET_SIZE)
                    img_path = base + ".png"
                    cv2.imwrite(img_path, face_img_qc)

                    # Imagen con malla original
                    face_with_mesh = face_img_qc.copy()
                    for (lx, ly) in lm_qc:
                        cv2.circle(face_with_mesh, (int(lx), int(ly)), 1, (0, 255, 0), -1)
                    img_mesh_path = base + "_mesh.png"
                    cv2.imwrite(img_mesh_path, face_with_mesh)

                    # Imagen warpeada a plantilla (si se pudo)
                    warped_path = None
                    if warped_img is not None:
                        warped_path = base + "_qc.png"
                        cv2.imwrite(warped_path, warped_img)

                    # Landmarks y metadatos (px + normalizados)
                    lm_path = base + "_landmarks_norm.npz"
                    np.savez(
                        lm_path,
                        landmarks_px=lm_qc,
                        landmarks_norm=lm_norm,
                        roi_size=(QC_TARGET_SIZE, QC_TARGET_SIZE),
                    )

                    print(f"[{mode.upper()}] Captura estable guardada:")
                    print(f"  Imagen:         {img_path}")
                    print(f"  Imagen + malla: {img_mesh_path}")
                    if warped_path is not None:
                        print(f"  Imagen QC:      {warped_path}")
                    print(f"  Landmarks+norm: {lm_path}")
                    print(f"  Frames estables: {stable_result['stable_frames']}")
                    print(f"  Capturas en esta sesión: {capture_idx}")

                    cv2.imshow("Captura estable con malla", face_with_mesh)
                    if warped_img is not None:
                        cv2.imshow("Captura warpeada QC", warped_img)

                    saved_count += 1

                    if stop_after_first_save:
                        should_stop = True

                else:
                    reason = decision.get("reason", "unknown")
                    if reason == "quota_reached":
                        print("Cuota de capturas alcanzada en esta sesion; no se guardan más.")
                        should_stop = True
                    else:
                        print(f"Captura estable NO guardada (razón: {reason}).")

                controller.reset()

            # 7.b Mensaje de login persistente (si lo hay)
            if mode == "login" and last_login_msg is not None:
                cv2.putText(
                    frame_display,
                    last_login_msg,
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    last_login_color,
                    2,
                    cv2.LINE_AA,
                )

            # 8. Mostrar frame
            cv2.imshow(win_title, frame_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Finalizando ejecucion.")
                break

            if should_stop:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Sesión '{mode}' finalizada. Capturas guardadas: {saved_count}")
