from typing import Tuple

import cv2
import numpy as np

from canonical_template import CanonicalFaceTemplate


class QuasiConformalWarper:
    """
    Warper cuasiconformal discreto (piecewise afín por triángulo).

    Dado:
      - una imagen de rostro (src_img) en un dominio cuadrado HxW,
      - landmarks de ese rostro (src_landmarks),
      - una plantilla facial canónica (CanonicalFaceTemplate),

    construye un warpeo que lleva los triángulos fuente a los triángulos
    destino usando transformaciones afines por triángulo.

    Esto es una forma de interpolación basada en landmarks + triangulación
    (LBS: Landmark-Based Scattered interpolation) sobre el mallado Delaunay.

    Permite un warpeo PARCIAL controlado por alpha:

      - alpha = 0.0 -> no deforma (identidad).
      - alpha = 1.0 -> lleva completamente a la plantilla.
      - 0 < alpha < 1 -> mezcla entre forma original y plantilla.
    """

    def __init__(
        self,
        template: CanonicalFaceTemplate,
        alpha: float = 1.0,
        background_value: int = 0,
    ) -> None:
        """
        :param template: instancia de CanonicalFaceTemplate.
        :param alpha: factor de mezcla [0,1] entre landmarks fuente y plantilla.
        :param background_value: valor de fondo (0=negro, 255=blanco).
        """
        self.template = template
        self.size = int(template.img_size)

        # Clampear alpha a [0,1]
        alpha = float(alpha)
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0
        self.alpha = alpha

        # Fondo clippeado
        self.background_value = int(np.clip(background_value, 0, 255))


    def _ensure_size_and_scale_landmarks(
        self, src_img: np.ndarray, src_landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asegura que la imagen y los landmarks estén en el mismo tamaño
        que la plantilla (self.size x self.size).
        Si no lo están, reescala ambos.
        """
        if src_img is None or src_img.size == 0:
            raise ValueError("src_img está vacío.")

        h, w, _ = src_img.shape
        if h == self.size and w == self.size:
            return src_img, src_landmarks.astype(np.float32)

        scale_x = self.size / float(w)
        scale_y = self.size / float(h)

        img_resized = cv2.resize(src_img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        lm_rescaled = src_landmarks.astype(np.float32).copy()
        lm_rescaled[:, 0] *= scale_x
        lm_rescaled[:, 1] *= scale_y

        return img_resized, lm_rescaled

    @staticmethod
    def _affine_jacobian(src_tri: np.ndarray, dst_tri: np.ndarray) -> np.ndarray:
        """
        Calcula el Jacobiano 2x2 (matriz afín sin traslación) que lleva
        src_tri (3 puntos) a dst_tri (3 puntos).

        src_tri, dst_tri: arrays (3,2) con (x,y) y (u,v) respectivamente.

        Retorna:
            J: np.ndarray shape (2,2)
        """
        # Construimos el sistema:
        # [x y 1] [a11] = [u]
        #         [a12]
        #         [t1 ]
        #
        # [x y 1] [a21] = [v]
        #         [a22]
        #         [t2 ]
        X = np.hstack([src_tri, np.ones((3, 1), dtype=np.float32)])  # (3,3)
        U = dst_tri[:, 0:1]  # (3,1)
        V = dst_tri[:, 1:2]  # (3,1)

        # Resolvemos en mínimos cuadrados (por robustez):
        A_u, _, _, _ = np.linalg.lstsq(X, U, rcond=None)  # (3,1)
        A_v, _, _, _ = np.linalg.lstsq(X, V, rcond=None)  # (3,1)

        a11, a12, _ = A_u.flatten()
        a21, a22, _ = A_v.flatten()

        J = np.array([[a11, a12],
                      [a21, a22]], dtype=np.float64)
        return J

    @staticmethod
    def _beltrami_from_jacobian(J: np.ndarray) -> complex:
        """
        Dado un Jacobiano 2x2:

            J = [[u_x, u_y],
                 [v_x, v_y]]

        calcula f_z y f_{\bar z} y devuelve el coeficiente de Beltrami:

            mu = f_{\bar z} / f_z

        Si |f_z| es muy pequeño, devuelve 0 (evita división por 0).
        """
        u_x, u_y = J[0, 0], J[0, 1]
        v_x, v_y = J[1, 0], J[1, 1]

        # Según la definición estándar:
        # f_z     = 0.5 * [(u_x + v_y) + i (v_x - u_y)]
        # f_zbar  = 0.5 * [(u_x - v_y) + i (v_x + u_y)]
        f_z = 0.5 * ((u_x + v_y) + 1j * (v_x - u_y))
        f_zbar = 0.5 * ((u_x - v_y) + 1j * (v_x + u_y))

        if np.abs(f_z) < 1e-8:
            return 0.0 + 0.0j

        mu = f_zbar / f_z
        return mu

    # Warpeo LBS
    def warp(self, src_img: np.ndarray, src_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica el warpeo piecewise-afín desde src_img/src_landmarks hacia la plantilla,
        con mezcla controlada por alpha.

        :param src_img: imagen de entrada (H, W, 3).
        :param src_landmarks: np.ndarray (N, 2) en píxeles sobre src_img.
        :return:
            warped_img: imagen (size, size, 3) deformada.
            warped_landmarks: np.ndarray (N, 2) en píxeles en el dominio destino
                              (landmarks mezclados según alpha).
        """
        # Ajustar tamaño
        src_img, src_landmarks = self._ensure_size_and_scale_landmarks(src_img, src_landmarks)

        # Imagen destino inicializada al valor de fondo
        dst_img = np.full_like(src_img, self.background_value, dtype=src_img.dtype)

        canonical_landmarks = self.template.landmarks_px.astype(np.float32)  # (N, 2)
        triangles = self.template.triangles  # (M, 3)

        # Landmarks destino mezclados (warp parcial)
        # alpha=0 -> sólo src_landmarks, alpha=1 -> sólo plantilla
        blended_landmarks = (1.0 - self.alpha) * src_landmarks.astype(np.float32) + \
                            self.alpha * canonical_landmarks

        for tri in triangles:
            i0, i1, i2 = tri
            src_tri = np.float32([
                src_landmarks[i0],
                src_landmarks[i1],
                src_landmarks[i2],
            ])
            dst_tri = np.float32([
                blended_landmarks[i0],
                blended_landmarks[i1],
                blended_landmarks[i2],
            ])

            # Rectángulos bounding box
            r_src = cv2.boundingRect(src_tri)
            r_dst = cv2.boundingRect(dst_tri)

            x_src, y_src, w_src, h_src = r_src
            x_dst, y_dst, w_dst, h_dst = r_dst

            if w_src <= 0 or h_src <= 0 or w_dst <= 0 or h_dst <= 0:
                continue

            # Coordenadas de triángulo relativas a su bbox
            src_tri_rect = src_tri.copy()
            src_tri_rect[:, 0] -= x_src
            src_tri_rect[:, 1] -= y_src

            dst_tri_rect = dst_tri.copy()
            dst_tri_rect[:, 0] -= x_dst
            dst_tri_rect[:, 1] -= y_dst

            # Recortes de imagen fuente y destino
            src_cropped = src_img[y_src: y_src + h_src, x_src: x_src + w_src]
            if src_cropped.size == 0:
                continue

            # Matriz afín (2x3) que lleva src_tri_rect -> dst_tri_rect
            M = cv2.getAffineTransform(src_tri_rect, dst_tri_rect)

            # Warpear el parche fuente al espacio del triángulo destino
            warped_patch = cv2.warpAffine(
                src_cropped,
                M,
                (w_dst, h_dst),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Máscara del triángulo destino
            mask = np.zeros((h_dst, w_dst, 1), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(dst_tri_rect), 1.0, lineType=cv2.LINE_AA)
            mask_3 = np.repeat(mask, 3, axis=2)

            # Región de interés en la imagen destino
            dst_roi = dst_img[y_dst: y_dst + h_dst, x_dst: x_dst + w_dst]
            if dst_roi.shape[:2] != warped_patch.shape[:2]:
                continue

            # Combinación: donde mask=1, ponemos warped_patch; donde 0, mantenemos dst_roi
            dst_roi[:] = dst_roi * (1.0 - mask_3) + warped_patch * mask_3
            dst_img[y_dst: y_dst + h_dst, x_dst: x_dst + w_dst] = dst_roi

        # Los landmarks destino son los blended (después de warp parcial)
        warped_landmarks = blended_landmarks.copy()
        return dst_img, warped_landmarks


    # Calculo beltrami por triangulo
    def compute_beltrami(self, src_landmarks: np.ndarray) -> np.ndarray:
        """
        Calcula el coeficiente de Beltrami μ_k para cada triángulo de la malla,
        correspondiente al mapeo LBS desde src_landmarks hacia los
        landmarks mezclados (blended_landmarks).

        IMPORTANTE:
          - Asume que src_landmarks están en el mismo sistema de coordenadas
            que la plantilla (ej. 256x256).
          - Si no estás seguro, pásalos antes por el mismo resize que usas
            en el pipeline (face_img_qc / QC_TARGET_SIZE).

        :param src_landmarks: np.ndarray (N,2) con (x,y) en dominio fuente.
        :return: np.ndarray (num_triangles,) de dtype complejo, con μ por triángulo.
        """
        src_landmarks = src_landmarks.astype(np.float32)

        canonical_landmarks = self.template.landmarks_px.astype(np.float32)  # (N, 2)
        triangles = self.template.triangles  # (M, 3)

        # Landmarks destino mezclados (igual que en warp)
        blended_landmarks = (1.0 - self.alpha) * src_landmarks + \
                            self.alpha * canonical_landmarks

        mus = []

        for tri in triangles:
            i0, i1, i2 = tri
            src_tri = src_landmarks[np.array([i0, i1, i2], dtype=int), :]
            dst_tri = blended_landmarks[np.array([i0, i1, i2], dtype=int), :]

            # Descartamos triángulos degenerados (área ~ 0)
            area = 0.5 * np.abs(
                (src_tri[1, 0] - src_tri[0, 0]) * (src_tri[2, 1] - src_tri[0, 1]) -
                (src_tri[2, 0] - src_tri[0, 0]) * (src_tri[1, 1] - src_tri[0, 1])
            )
            if area < 1e-6:
                mus.append(0.0 + 0.0j)
                continue

            J = self._affine_jacobian(src_tri, dst_tri)
            mu = self._beltrami_from_jacobian(J)
            mus.append(mu)

        return np.array(mus, dtype=np.complex128)
