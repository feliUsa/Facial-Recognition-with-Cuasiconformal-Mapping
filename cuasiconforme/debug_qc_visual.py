import os
import cv2
import numpy as np

from canonical_template import CanonicalFaceTemplate


BASE_DIR = (
    "/home/daniel/Universidad/experimentosLibrerias/cuasiconforme/"
    "capturas_yunet_facemesh/register"
)

TEMPLATE_PATH = "canonical_template.npz"


USER1 = "daniel"
BASE1 = "register_20251204_131531_1"

#USER2 = "daniel"
#BASE2 = "register_20251203_211858_3"

USER2 = "juanPablo"
BASE2 = "offline_juanPablo_1_20251204_130407"


def load_sample(user_name: str, base_name: str):
    """
    Carga:
      - imagen original normalizada (256x256): base.png
      - imagen warpeada QC: base_qc.png

    Estructura esperada:
      BASE_DIR/
        user_name/
          base_name.png
          base_name_qc.png
    """
    user_dir = os.path.join(BASE_DIR, user_name)

    raw_path = os.path.join(user_dir, base_name + ".png")
    qc_path = os.path.join(user_dir, base_name + "_qc.png")

    img_raw = cv2.imread(raw_path)
    img_qc = cv2.imread(qc_path)

    if img_raw is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen RAW: {raw_path}")
    if img_qc is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen QC:  {qc_path}")

    return img_raw, img_qc


def draw_canonical_mesh(img: np.ndarray, template: CanonicalFaceTemplate) -> np.ndarray:
    """
    Dibuja la malla canónica (triángulos + puntos) sobre la imagen QC.
    Se asume que:
      - img tiene el mismo tamaño que template.img_size (ej. 256x256).
      - La imagen QC está ya en el dominio de la plantilla.
    """
    overlay = img.copy()
    lm = template.get_landmarks()          # (N, 2)
    triangles = template.get_triangles()   # (M, 3)

    # Dibujar triángulos
    for tri in triangles:
        i0, i1, i2 = tri
        p0 = tuple(np.round(lm[i0]).astype(int))
        p1 = tuple(np.round(lm[i1]).astype(int))
        p2 = tuple(np.round(lm[i2]).astype(int))
        cv2.line(overlay, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(overlay, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(overlay, p2, p0, (0, 255, 0), 1, cv2.LINE_AA)

    # Dibujar puntos
    for (x, y) in lm:
        cv2.circle(overlay, (int(x), int(y)), 1, (0, 0, 255), -1)

    return overlay


def stack_images_grid(img11, img12, img21, img22, resize_to=None):
    """
    Crea una cuadrícula 2x2:
      [ img11 | img12 ]
      [ img21 | img22 ]
    Opcionalmente reescala todas al mismo tamaño.
    """
    imgs = [img11, img12, img21, img22]

    if resize_to is not None:
        imgs = [cv2.resize(im, resize_to) for im in imgs]

    row1 = np.hstack((imgs[0], imgs[1]))
    row2 = np.hstack((imgs[2], imgs[3]))
    grid = np.vstack((row1, row2))
    return grid


def main():
    # Cargar plantilla canónica
    print(f"Cargando plantilla desde {TEMPLATE_PATH}")
    template = CanonicalFaceTemplate.load(TEMPLATE_PATH)

    # Cargar dos ejemplos (A y B)
    print(f"Cargando ejemplos:")
    print(f"  Usuario1={USER1}, Base1={BASE1}")
    print(f"  Usuario2={USER2}, Base2={BASE2}")

    raw1, qc1 = load_sample(USER1, BASE1)
    raw2, qc2 = load_sample(USER2, BASE2)

    # Asegurar que tienen el mismo tamaño que la plantilla
    size = template.img_size
    raw1 = cv2.resize(raw1, (size, size))
    raw2 = cv2.resize(raw2, (size, size))
    qc1 = cv2.resize(qc1, (size, size))
    qc2 = cv2.resize(qc2, (size, size))

    # Dibujar malla canónica sobre las QC
    qc1_mesh = draw_canonical_mesh(qc1, template)
    qc2_mesh = draw_canonical_mesh(qc2, template)

    # Crear grid 2x2:
    #   [ raw1      | raw2      ]
    #   [ qc1_mesh  | qc2_mesh  ]
    grid = stack_images_grid(raw1, raw2, qc1_mesh, qc2_mesh)

    cv2.imshow("QC Debug: Arriba=RAW, Abajo=QC+Plantilla", grid)
    print("Ventana abierta. Pulsa cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
