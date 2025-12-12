import numpy as np


class StableCaptureController:
    """
    Controlador de captura estable de rostro.

    Responsabilidades:
    - Llevar un contador de frames consecutivos con detección válida (ROI + landmarks).
    - Cuando se alcanza un umbral (ej. 20 frames), devolver una "captura estable"
      lista para ser usada en el mapeo cuasiconformal.

    Métodos principales:
    - update(has_detection, face_roi, landmarks) -> dict o None
    - reset() -> reinicia el contador para un nuevo usuario.
    """

    def __init__(self, required_stable_frames: int = 20) -> None:
        self.required_stable_frames = required_stable_frames
        self._counter = 0
        self._captured = False
        self._sum_landmarks = None
        self._valid_count = 0
        self._last_roi = None

    def reset(self) -> None:
        self._counter = 0
        self._captured = False
        self._sum_landmarks = None
        self._valid_count = 0
        self._last_roi = None

    @property
    def current_stable_frames(self) -> int:
        return self._counter

    def update(self, has_detection: bool, face_roi: np.ndarray, landmarks: np.ndarray):
        if (
            not has_detection
            or face_roi is None
            or landmarks is None
            or face_roi.size == 0
            or landmarks.size == 0
        ):
            self.reset()
            return None

        self._counter += 1
        self._last_roi = face_roi.copy()

        if self._sum_landmarks is None:
            self._sum_landmarks = np.zeros_like(landmarks)
        self._sum_landmarks += landmarks
        self._valid_count += 1

        if self._captured:
            return None

        if self._counter >= self.required_stable_frames:
            mean_landmarks = self._sum_landmarks / max(1, self._valid_count)
            self._captured = True
            return {
                "face_image": self._last_roi,
                "landmarks": mean_landmarks,
                "stable_frames": self._counter,
            }

        return None