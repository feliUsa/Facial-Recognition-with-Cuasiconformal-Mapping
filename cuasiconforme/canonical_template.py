import os
from typing import List

import cv2
import numpy as np

from yunet_detector import YuNetFaceDetector
from facemesh_helper import FaceMeshHelper


def _build_delaunay_triangles(points: np.ndarray, img_size: int) -> np.ndarray:
    """
    Construye una triangulación de Delaunay 2D usando OpenCV Subdiv2D.

    :param points: np.ndarray (N, 2) con coords (x, y) en píxeles.
    :param img_size: tamaño de la imagen cuadrada (ej. 256).
    :return: np.ndarray (M, 3) con índices de triángulos.
    """
    points = points.astype(np.float32)
    h = w = int(img_size)

    # Rectangulo que cubre todo el dominio
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # Insertar puntos
    for (x, y) in points:
        subdiv.insert((float(x), float(y)))

    triangle_list = subdiv.getTriangleList()
    if triangle_list is None or len(triangle_list) == 0:
        raise RuntimeError("No se pudieron obtener triangulos de Delaunay.")

    triangles = []
    used = set()

    for t in triangle_list:
        x1, y1, x2, y2, x3, y3 = t
        tri_pts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)

        # Filtrar triangulos fuera de la imagen
        if np.any(tri_pts[:, 0] < 0) or np.any(tri_pts[:, 0] >= w):
            continue
        if np.any(tri_pts[:, 1] < 0) or np.any(tri_pts[:, 1] >= h):
            continue

        # Para cada vertice del triángulo, encontrar el índice más cercano en 'points'
        idxs = []
        for (vx, vy) in tri_pts:
            diffs = points - np.array([vx, vy], dtype=np.float32)
            d2 = np.sum(diffs * diffs, axis=1)
            j = int(np.argmin(d2))
            idxs.append(j)

        # Evitar triangulos degenerados (vertices repetidos)
        if len(set(idxs)) < 3:
            continue

        tri_idx = tuple(sorted(idxs))
        if tri_idx in used:
            continue
        used.add(tri_idx)
        triangles.append(list(tri_idx))

    if not triangles:
        raise RuntimeError("Triangulacion vacía; revisa tus landmarks.")

    return np.array(triangles, dtype=np.int32)


class CanonicalFaceTemplate:
    """
    Plantilla facial canónica para el mapeo cuasiconformal discreto.

    Guarda:
    - landmarks_canónicos (en píxeles, dominio cuadrado img_size x img_size).
    - triangulación fija (índices a esos landmarks).
    """

    def __init__(self, landmarks_px: np.ndarray, triangles: np.ndarray, img_size: int) -> None:
        """
        :param landmarks_px: np.ndarray (N, 2) en píxeles dentro de [0, img_size).
        :param triangles: np.ndarray (M, 3) con índices de triángulos.
        :param img_size: tamaño de la imagen cuadrada (ej. 256).
        """
        self.landmarks_px = landmarks_px.astype(np.float32)
        self.triangles = triangles.astype(np.int32)
        self.img_size = int(img_size)

    # CONSTRUCTOR SIMPLE
    @classmethod
    def from_landmarks_px(cls, landmarks_px: np.ndarray, img_size: int) -> "CanonicalFaceTemplate":
        """
        Construye la plantilla usando directamente una malla de landmarks
        (por ejemplo, de una captura estable).

        :param landmarks_px: np.ndarray (N, 2) en píxeles.
        :param img_size: tamaño de la imagen cuadrada.
        """
        if landmarks_px is None or landmarks_px.ndim != 2 or landmarks_px.shape[1] != 2:
            raise ValueError("landmarks_px debe tener shape (N, 2).")

        # Asegurar que los landmarks están dentro del rango
        lm = landmarks_px.copy().astype(np.float32)
        lm[:, 0] = np.clip(lm[:, 0], 0, img_size - 1)
        lm[:, 1] = np.clip(lm[:, 1], 0, img_size - 1)

        triangles = _build_delaunay_triangles(lm, img_size)
        return cls(lm, triangles, img_size)

    # Construir plantilla canonica
    @classmethod
    def build_from_dataset(
        cls,
        dataset_root: str,
        yunet_model_path: str,
        img_size: int = 256,
        max_images_per_person: int = 40,
        max_total_images: int = 2000,
        min_face_size: int = 80,
        asymmetry_threshold: float = 0.25,
    ) -> "CanonicalFaceTemplate":
        """
        Recorre un dataset de rostros (subcarpetas por persona), detecta caras,
        extrae landmarks con FaceMesh y construye una plantilla canónica
        como el promedio de todas las mallas normalizadas.

        - dataset_root: carpeta raíz del dataset (subcarpetas por actor/actriz).
        - yunet_model_path: ruta al modelo ONNX de YuNet.
        - img_size: tamaño canónico para la ROI (ej. 256).
        - max_images_per_person: máximo de imágenes a usar por carpeta/persona.
        - max_total_images: tope global para no procesar miles de imágenes.
        - min_face_size: mínimo tamaño (en píxeles) del lado menor de la ROI.
        - asymmetry_threshold: umbral para is_frontal_enough de FaceMesh.
        """
        dataset_root = os.path.abspath(dataset_root)
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_root}")

        print(f"[CT] Dataset raíz: {dataset_root}")
        print(f"[CT] Modelo YuNet: {yunet_model_path}")
        print(f"[CT] Tamaño canónico (img_size): {img_size}")

        # Instanciar detector y FaceMesh para imágenes estáticas
        detector = YuNetFaceDetector(yunet_model_path)
        fm_helper = FaceMeshHelper(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )

        landmarks_norm_list: List[np.ndarray] = []
        images_per_person = {}
        total_used = 0

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for root, dirs, files in os.walk(dataset_root):
            # Nombre de la persona (carpeta)
            person_id = os.path.basename(root)
            if person_id == os.path.basename(dataset_root):
                # raíz, no una persona
                continue

            used_for_this_person = images_per_person.get(person_id, 0)
            if used_for_this_person >= max_images_per_person:
                continue

            for fname in files:
                if total_used >= max_total_images:
                    break

                ext = os.path.splitext(fname)[1].lower()
                if ext not in valid_exts:
                    continue

                if used_for_this_person >= max_images_per_person:
                    break

                img_path = os.path.join(root, fname)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[CT] Aviso: no se pudo leer {img_path}")
                    continue

                h, w, _ = img.shape

                # 1) Detección principal con YuNet
                boxes, best_box = detector.detect(img)
                bbox = None

                if best_box is not None:
                    bbox = best_box.astype(int)
                else:
                    # Fallback con FaceMesh en frame completo
                    fb = fm_helper.detect_face_fallback_bbox(img)
                    if fb is not None:
                        bbox = fb.astype(int)

                if bbox is None:
                    # No hay cara, descartamos
                    continue

                x1, y1, x2, y2 = bbox
                # Clamp
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                face_roi = img[y1:y2, x1:x2]
                if face_roi is None or face_roi.size == 0:
                    continue

                h_roi, w_roi, _ = face_roi.shape
                if min(h_roi, w_roi) < min_face_size:
                    # Cara demasiado pequeña, descartamos
                    continue

                # 2) Landmarks sobre la ROI
                lm_roi = fm_helper.extract_landmarks_from_roi(face_roi)
                if lm_roi is None:
                    continue

                # 3) Pose frontal suficiente
                if not fm_helper.is_frontal_enough(lm_roi, asymmetry_threshold):
                    continue

                # 4) Reescalar ROI a img_size x img_size y escalar landmarks
                scale_x = img_size / float(w_roi)
                scale_y = img_size / float(h_roi)

                lm_qc = lm_roi.astype(np.float32).copy()
                lm_qc[:, 0] *= scale_x
                lm_qc[:, 1] *= scale_y

                # 5) Normalizar a [0,1] para promediar formas
                lm_norm = lm_qc / float(img_size)
                landmarks_norm_list.append(lm_norm)

                used_for_this_person += 1
                images_per_person[person_id] = used_for_this_person
                total_used += 1

                if total_used % 50 == 0:
                    print(f"Procesadas {total_used} imágenes válidas...")

                if total_used >= max_total_images:
                    break

            if total_used >= max_total_images:
                break

        if not landmarks_norm_list:
            raise RuntimeError("No se obtuvieron landmarks válidos del dataset.")

        print(f"Total de imágenes usadas: {total_used}")
        print(f"Numero de personas con muestras: {len(images_per_person)}")

        # Promedio de formas
        stack = np.stack(landmarks_norm_list, axis=0)  # (K, 468, 2)
        mean_landmarks_norm = np.mean(stack, axis=0)   # (468, 2)

        # Pasar a píxeles
        canonical_landmarks_px = mean_landmarks_norm * float(img_size)

        # Construir plantilla (triangulación + almacenaje)
        return cls.from_landmarks_px(canonical_landmarks_px, img_size)

    # ---------- UTILIDADES BÁSICAS ----------

    def save(self, path: str) -> None:
        """
        Guarda la plantilla en disco (NPZ).
        """
        np.savez(
            path,
            landmarks_px=self.landmarks_px,
            triangles=self.triangles,
            img_size=self.img_size,
        )

    @classmethod
    def load(cls, path: str) -> "CanonicalFaceTemplate":
        """
        Carga una plantilla previamente guardada.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontro la plantilla canonica en: {path}")

        data = np.load(path)
        landmarks_px = data["landmarks_px"]
        triangles = data["triangles"]
        img_size = int(data["img_size"])
        return cls(landmarks_px, triangles, img_size)

    def get_landmarks(self) -> np.ndarray:
        return self.landmarks_px.copy()

    def get_triangles(self) -> np.ndarray:
        return self.triangles.copy()


# Construir template
if __name__ == "__main__":
    DATASET_ROOT = (
        "/home/daniel/Universidad/experimentosLibrerias/"
        "cuasiconforme/datasetCreatingCanonicalTemplate/Celebrity Faces Dataset/"
    )

    YUNET_MODEL_PATH = (
        "/home/daniel/Universidad/experimentosLibrerias/identificacion/"
        "modelos/face_detection_yunet_2023mar.onnx"
    )

    OUTPUT_TEMPLATE_PATH = "canonical_template.npz"
    IMG_SIZE = 256

    print("Construyendo plantilla canonica desde dataset de celebridades...")
    template = CanonicalFaceTemplate.build_from_dataset(
        dataset_root=DATASET_ROOT,
        yunet_model_path=YUNET_MODEL_PATH,
        img_size=IMG_SIZE,
        max_images_per_person=40,
        max_total_images=2000,
        min_face_size=80,
        asymmetry_threshold=0.25,
    )

    template.save(OUTPUT_TEMPLATE_PATH)
    print(f"Plantilla canonica guardada en: {OUTPUT_TEMPLATE_PATH}")
