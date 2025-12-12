import cv2
import numpy as np
import mediapipe as mp


class FaceMeshHelper:
    """
    Helper para MediaPipe FaceMesh.

    Responsabilidades:
    - Extraer 468 landmarks 2D sobre una ROI de cara (BGR).
    - Usar FaceMesh sobre el frame completo como fallback de detección
      cuando YuNet no detecta ninguna cara.
    - Evaluar si la pose es suficientemente frontal para considerar el frame estable.

    Métodos principales:
    - extract_landmarks_from_roi(face_roi_bgr) -> np.ndarray (468, 2) o None
    - detect_face_fallback_bbox(frame_bgr) -> np.ndarray(4,) [x1, y1, x2, y2] o None
    - is_frontal_enough(landmarks_2d, asymmetry_threshold) -> bool
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """
        Inicializa el modelo FaceMesh.

        :param static_image_mode: True para tratar cada frame como imagen independiente.
        :param max_num_faces: Máximo número de caras a detectar.
        :param min_detection_confidence: Umbral mínimo de detección.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
        )

    def extract_landmarks_from_roi(self, face_roi_bgr):
        """
        Extrae los 468 landmarks de FaceMesh en una ROI de cara (BGR).

        :param face_roi_bgr: Recorte BGR de la cara.
        :return: np.ndarray shape (468, 2) con coords (x, y) en píxeles relativas a la ROI,
                 o None si no se detecta rostro.
        """
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return None

        h, w, _ = face_roi_bgr.shape
        rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]

        landmarks_2d = np.stack([np.array(xs), np.array(ys)], axis=1)  # (468, 2)
        return landmarks_2d

    def detect_face_fallback_bbox(self, frame_bgr):
        """
        Usa FaceMesh sobre el frame completo para obtener una bounding box
        cuando YuNet no detecta ninguna cara.

        :param frame_bgr: Frame completo en BGR.
        :return: np.ndarray shape (4,) con [x1, y1, x2, y2] o None si no hay cara.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]

        x1 = max(0.0, float(min(xs)))
        y1 = max(0.0, float(min(ys)))
        x2 = min(float(w - 1), float(max(xs)))
        y2 = min(float(h - 1), float(max(ys)))

        if x2 <= x1 or y2 <= y1:
            return None

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def is_frontal_enough(self, landmarks_2d: np.ndarray, asymmetry_threshold: float = 0.3) -> bool:
        """
        Evalúa si la cara es suficientemente frontal usando asimetría nariz–ojos.

        Heurístico:
        - Tomamos:
            índice 1   -> punta de la nariz
            índice 33  -> esquina externa ojo derecho
            índice 263 -> esquina externa ojo izquierdo
        - Calculamos distancias horizontales nariz↔ojo_izq y nariz↔ojo_der.
        - Si la diferencia relativa entre esas distancias es pequeña, asumimos
          que la cara está más o menos frontal.

        :param landmarks_2d: np.ndarray (468, 2) con coords en píxeles.
        :param asymmetry_threshold: Umbral máximo de asimetría aceptable (0 = perfecto).
        :return: True si la pose es suficientemente frontal, False si está muy de perfil.
        """
        if landmarks_2d is None or landmarks_2d.shape[0] <= 263:
            return False

        try:
            nose = landmarks_2d[1]      # (x, y)
            right_outer = landmarks_2d[33]
            left_outer = landmarks_2d[263]
        except IndexError:
            return False

        d_left = abs(nose[0] - left_outer[0])
        d_right = abs(nose[0] - right_outer[0])

        denom = d_left + d_right
        if denom < 1e-3:
            return False

        r = abs(d_left - d_right) / denom  # índice de asimetría

        return r <= asymmetry_threshold
