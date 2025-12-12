import numpy as np


class UserCaptureManager:
    """
    Gestiona las capturas por usuario usando sólo la malla de FaceMesh como firma.

    Responsabilidades:
    - Mantener una "sesión" de usuario actual con:
        - firma (signature) basada en landmarks 2D,
        - contador de capturas ya realizadas.
    - Decidir, para cada nueva captura estable (landmarks):
        - si pertenece al mismo usuario o a uno nuevo,
        - si se debe guardar (no superar max_captures_per_user).

    Uso típico:
        manager = UserCaptureManager(max_captures_per_user=5, distance_threshold=0.25)
        decision = manager.should_save(landmarks_2d)
        if decision["save"]:
            # guardas imagen + landmarks
        else:
            # ignoras esta captura (ya llegó a cuota para ese usuario)
    """

    def __init__(self, max_captures_per_user: int = 5, distance_threshold: float = 0.25) -> None:
        """
        :param max_captures_per_user: Número máximo de capturas permitidas para un mismo usuario.
        :param distance_threshold: Umbral de distancia coseno para considerar que dos firmas
                                   pertenecen al mismo usuario (0 = idéntico, valores pequeños ~ mismo).
        """
        self.max_captures_per_user = max_captures_per_user
        self.distance_threshold = distance_threshold

        self.current_signature = None  # np.ndarray o None
        self.current_captures = 0      # cuántas capturas lleva el usuario actual

    def reset_session(self) -> None:
        """
        Resetea la sesión de usuario actual (por ejemplo, cuando la persona se va).
        """
        self.current_signature = None
        self.current_captures = 0

    @property
    def captures_for_current_user(self) -> int:
        """
        Devuelve cuántas capturas lleva el usuario actual.
        """
        return self.current_captures

    def _build_signature(self, landmarks_2d: np.ndarray):
        """
        Construye una firma a partir de landmarks 2D (468, 2).

        Pasos:
        - Centrar la malla (restar el centroide).
        - Normalizar por escala (tamaño medio al centro).
        - Aplanar a vector 1D.
        - Normalizar L2 (para usar distancia coseno).

        :param landmarks_2d: np.ndarray (N, 2), típicamente (468, 2).
        :return: np.ndarray 1D (firma normalizada) o None si algo falla.
        """
        if landmarks_2d is None or landmarks_2d.ndim != 2 or landmarks_2d.shape[1] != 2:
            return None

        pts = landmarks_2d.astype(np.float32)

        # Centrar en el centroide
        center = pts.mean(axis=0)  # (2,)
        pts_centered = pts - center

        # Escala: distancia RMS al centro
        dists = np.sqrt(np.sum(pts_centered ** 2, axis=1))  # (N,)
        scale = float(np.mean(dists))
        if scale < 1e-6:
            return None

        pts_norm = pts_centered / scale  # (N, 2)

        # Aplanar
        vec = pts_norm.flatten().astype(np.float32)  # (2N,)

        # Normalizar L2 → firma lista para distancia coseno
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None

        signature = vec / norm
        return signature

    @staticmethod
    def _cosine_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Distancia coseno entre dos firmas normalizadas:
        dist = 1 - dot(sig1, sig2)

        - 0  -> vectores idénticos
        - ~0.2..0.4 -> similares
        - ~1        -> ortogonales
        - >1        -> bastante distintos

        :return: float
        """
        if sig1 is None or sig2 is None:
            return float("inf")

        dot = float(np.dot(sig1, sig2))
        # Como están normalizados, no hace falta dividir por normas.
        # Aseguramos rango [-1, 1] por seguridad numérica:
        dot = max(min(dot, 1.0), -1.0)
        return 1.0 - dot

    def should_save(self, landmarks_2d: np.ndarray):
        """
        Decide si se debe guardar una nueva captura para el usuario actual.

        Lógica:
        - Construye una firma 'signature' a partir de los landmarks.
        - Si no hay firma actual (no hay sesión):
            - inicia nueva sesión,
            - current_signature = signature,
            - current_captures = 1,
            - save = True, is_new_user = True.
        - Si ya hay firma actual:
            - calcula distancia coseno entre signature y current_signature.
            - Si distancia <= distance_threshold → mismo usuario:
                - si current_captures >= max_captures_per_user:
                    save = False (cuota alcanzada).
                - si no, incrementa current_captures, save = True.
                - opcional: actualiza current_signature con media de firmas.
            - Si distancia > distance_threshold → nuevo usuario:
                - reinicia firma y contador con esta captura,
                - save = True, is_new_user = True.

        :param landmarks_2d: np.ndarray (468, 2) de la captura estable.
        :return: dict con claves:
                 - "save": bool
                 - "is_new_user": bool
                 - "captures_for_user": int
                 - "distance": float o None
                 - "reason": str (opcional cuando save=False)
        """
        sig = self._build_signature(landmarks_2d)
        if sig is None:
            # Algo fue mal con la generación de firma; por seguridad no guardamos.
            return {
                "save": False,
                "is_new_user": False,
                "captures_for_user": self.current_captures,
                "distance": None,
                "reason": "invalid_signature",
            }

        # Caso 1: no hay usuario activo
        if self.current_signature is None:
            self.current_signature = sig
            self.current_captures = 1
            return {
                "save": True,
                "is_new_user": True,
                "captures_for_user": self.current_captures,
                "distance": 0.0,
            }

        # Caso 2: ya hay usuario activo → comparamos firmas
        dist = self._cosine_distance(self.current_signature, sig)

        if dist <= self.distance_threshold:
            # Mismo usuario
            if self.current_captures >= self.max_captures_per_user:
                # Cuota alcanzada: NO guardamos
                return {
                    "save": False,
                    "is_new_user": False,
                    "captures_for_user": self.current_captures,
                    "distance": dist,
                    "reason": "quota_reached",
                }

            # Todavía hay cupo → guardamos y actualizamos firma (opcional: media)
            self.current_captures += 1
            # Mezcla simple de firmas para robustez
            alpha = 0.5
            self.current_signature = (alpha * self.current_signature + (1 - alpha) * sig)
            # Re-normalizar
            norm = float(np.linalg.norm(self.current_signature))
            if norm > 1e-6:
                self.current_signature /= norm

            return {
                "save": True,
                "is_new_user": False,
                "captures_for_user": self.current_captures,
                "distance": dist,
            }

        else:
            # Nuevo usuario detectado → reiniciar sesión con esta captura
            self.current_signature = sig
            self.current_captures = 1
            return {
                "save": True,
                "is_new_user": True,
                "captures_for_user": self.current_captures,
                "distance": dist,
            }
