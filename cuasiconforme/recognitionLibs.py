from typing import Optional

import cv2
import numpy as np


try:
    from deepface import DeepFace
    _HAS_DEEPFACE = True
except ImportError:
    DeepFace = None
    _HAS_DEEPFACE = False


class BaseFaceRecognizer:
    """
    Interfaz base para modelos de reconocimiento facial.

    Esta clase NO maneja galería ni BD; solo define cómo producir embeddings
    y calcular distancias. La lógica de 1:N (identificar usuario) la maneja
    FaceSystem usando estos embeddings.
    """

    def __init__(self, img_size: int = 256) -> None:
        self.img_size = int(img_size)
        # Umbral por defecto pensado para distancia coseno.
        # Cada subclase lo puede ajustar.
        self.default_threshold: float = 0.35

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Asegura tamaño y formato adecuados para el modelo:
        - Reescala a (img_size, img_size)
        - Convierte de BGR (OpenCV) a RGB
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Imagen vacía en preprocess().")

        img_resized = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        return img_rgb

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula distancia coseno 1 - cos_sim.
        """
        a = a.astype(np.float32).ravel()
        b = b.astype(np.float32).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        cos_sim = float(np.dot(a, b) / (na * nb))
        return 1.0 - cos_sim


    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Dada una imagen BGR, devuelve un embedding 1D (np.ndarray).
        """
        raise NotImplementedError


class DeepFaceRecognizer(BaseFaceRecognizer):
    
    """
    Wrapper ligero sobre DeepFace para usar modelos como ArcFace o SFace.

    Uso típico:
        rec = DeepFaceRecognizer(model_name="ArcFace")
        emb = rec.embed(img_bgr)
    """

    def __init__(
        self,
        model_name: str = "ArcFace",
        img_size: int = 256,
    ) -> None:
        super().__init__(img_size=img_size)
        self.model_name = model_name

        print(f"[DeepFaceRecognizer] Usando modelo '{model_name}' con DeepFace.represent")

        # Ajuste de umbral base según modelo (valores orientativos)
        if model_name.lower() == "arcface":
            self.default_threshold = 0.35
        elif model_name.lower() == "sface":
            self.default_threshold = 0.40
        else:
            self.default_threshold = 0.40

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Produce el embedding de la imagen usando DeepFace.represent en modo "en memoria".
        NO pasamos `model=` (tu error de antes), solo `model_name`.
        """
        img_rgb = self.preprocess(img_bgr)

        reps = DeepFace.represent(
            img_path=img_rgb,
            model_name=self.model_name,
            enforce_detection=False,
        )

        # reps es una lista de dicts, tomamos el primero
        if isinstance(reps, list) and len(reps) > 0:
            rep0 = reps[0]
        else:
            raise RuntimeError("DeepFace.represent no devolvió ningún embedding.")

        emb = rep0.get("embedding", None)
        if emb is None:
            raise RuntimeError("DeepFace.represent no devolvió 'embedding' en la respuesta.")

        emb_arr = np.array(emb, dtype=np.float32).ravel()
        return emb_arr



class BuffaloLRecognizer(BaseFaceRecognizer):
    """
    Wrapper para el modelo 'buffalo_l' de insightface.

    Requisitos de instalación (en tu env):
        pip install insightface onnxruntime

    Nota: este wrapper vuelve a detectar el rostro internamente con insightface.
    Como ya le mandas una ROI alineada y grande, normalmente detecta 1 cara
    sin problema.
    """

    def __init__(self, img_size: int = 256, det_size=(256, 256)) -> None:
        # img_size aquí no es crítico, porque insightface recibe la imagen completa.
        super().__init__(img_size=img_size)

        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as e:
            raise ImportError(
                "BuffaloLRecognizer requiere el paquete 'insightface'. "
                "Instala con: pip install insightface onnxruntime"
            ) from e

        self.FaceAnalysis = FaceAnalysis
        self.app = self.FaceAnalysis(name="buffalo_l")
        # ctx_id = -1 → solo CPU; 0 si tienes GPU y drivers correctos.
        self.app.prepare(ctx_id=-1, det_size=det_size)

        # Umbral orientativo para buffalo_l (distancia coseno)
        self.default_threshold = 0.40

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Imagen vacía en BuffaloLRecognizer.embed().")

        # insightface trabaja en BGR internamente (usa OpenCV), así que pasamos BGR
        faces = self.app.get(img_bgr)
        if not faces:
            raise RuntimeError("BuffaloLRecognizer: no se detectó ningún rostro en la imagen.")

        # Tomamos el primer rostro (tu ROI ya está centrada y limpia)
        face = faces[0]

        # En versiones recientes, el embedding normalizado se expone como 'normed_embedding'
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = face.embedding  # fallback

        emb_arr = np.asarray(emb, dtype=np.float32).ravel()
        return emb_arr


class FacenetVGGFace2Recognizer(BaseFaceRecognizer):
    """
    Wrapper para Facenet-PyTorch (InceptionResnetV1, pesos VGGFace2).

    Requisitos:
        pip install facenet-pytorch torch torchvision

    Usa umbral de distancia coseno ~0.8 (ajústalo empíricamente).
    """

    def __init__(self, img_size: int = 160, device: Optional[str] = None) -> None:
        # Facenet suele trabajar con 160x160
        super().__init__(img_size=img_size)

        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as e:
            raise ImportError(
                "FacenetVGGFace2Recognizer requiere 'torch' y 'facenet-pytorch'. "
                "Instala con: pip install facenet-pytorch torch torchvision"
            ) from e

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Umbral orientativo para distancia coseno con Facenet
        self.default_threshold = 0.80

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Imagen vacía en FacenetVGGFace2Recognizer.embed().")

        # Preprocesado base: resize + BGR→RGB
        img_rgb = self.preprocess(img_bgr)  # (H, W, 3) en RGB

        # Normalizar a [0,1] y pasar a tensor NCHW
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)

        tensor = self.torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with self.torch.no_grad():
            emb = self.model(tensor)  # (1, 512) normalmente

        emb_arr = emb.squeeze(0).cpu().numpy().astype(np.float32).ravel()
        return emb_arr
