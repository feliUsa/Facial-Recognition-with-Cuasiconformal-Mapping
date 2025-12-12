import cv2
import numpy as np


class YuNetFaceDetector:
    """
    Detector de rostros basado en YuNet (OpenCV).

    Responsabilidades:
    - Cargar el modelo ONNX de YuNet.
    - Detectar caras en un frame BGR.
    - Devolver todas las cajas detectadas y la mejor caja (por score).

    Métodos principales:
    - detect(frame_bgr): -> (boxes, best_box)
        boxes: np.ndarray (N, 4) con [x1, y1, x2, y2]
        best_box: np.ndarray (4,) o None si no hay detecciones
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        """
        Inicializa el detector YuNet.

        :param model_path: Ruta al modelo ONNX de YuNet.
        :param score_threshold: Umbral mínimo de score para aceptar detecciones.
        :param nms_threshold: Umbral de NMS interno de YuNet.
        :param top_k: Máximo número de detecciones.
        """
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        # Se inicializa con un tamaño dummy; se actualiza por frame con setInputSize.
        self.detector = cv2.FaceDetectorYN_create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def detect(self, frame_bgr):
        """
        Detecta rostros en un frame BGR.

        :param frame_bgr: Imagen en BGR (frame de cámara).
        :return:
            boxes: np.ndarray de shape (N, 4) con [x1, y1, x2, y2].
            best_box: np.ndarray shape (4,) con la mejor cara o None.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.empty((0, 4), dtype=np.float32), None

        h, w, _ = frame_bgr.shape
        self.detector.setInputSize((w, h))

        retval, faces = self.detector.detect(frame_bgr)

        if faces is None or len(faces) == 0:
            return np.empty((0, 4), dtype=np.float32), None

        boxes = []
        scores = []

        # faces: (N, 15) → [x, y, w, h, score, l0x, l0y, ...]
        for f in faces:
            x, y, bw, bh = f[0:4]
            score = float(f[4])

            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = x1 + max(0.0, float(bw))
            y2 = y1 + max(0.0, float(bh))

            x2 = min(float(w - 1), x2)
            y2 = min(float(h - 1), y2)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)

        if not boxes:
            return np.empty((0, 4), dtype=np.float32), None

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        best_idx = int(np.argmax(scores))
        best_box = boxes[best_idx].copy()

        return boxes, best_box
