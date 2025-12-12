import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import psutil

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)

from recognitionLibs import (
    BuffaloLRecognizer,
    FacenetVGGFace2Recognizer,
    DeepFaceRecognizer,
)

# Ruta a tus registros (usuarios ya enrolados)
REGISTER_ROOT = os.path.join("capturas_yunet_facemesh", "register")


def build_gallery(
    register_root: str,
    recognizer,
) -> Dict[str, np.ndarray]:
    """
    Construye la galería 1:N para un modelo.
    Devuelve: {usuario: matriz (K, D) con embeddings de sus imágenes}.
    """
    gallery: Dict[str, np.ndarray] = {}

    if not os.path.isdir(register_root):
        print(f"[GAL] register_root '{register_root}' no existe.")
        return gallery

    users = [
        d for d in os.listdir(register_root)
        if os.path.isdir(os.path.join(register_root, d))
    ]
    if not users:
        print("[GAL] No hay usuarios en register_root.")
        return gallery

    for user in users:
        user_dir = os.path.join(register_root, user)

        # Preferimos las imágenes *_qc.png (warpeadas)
        files_qc = sorted(
            f for f in os.listdir(user_dir)
            if f.endswith("_qc.png")
        )
        if files_qc:
            files = [os.path.join(user_dir, f) for f in files_qc]
        else:
            # fallback a imágenes limpias
            files_clean = sorted(
                f for f in os.listdir(user_dir)
                if f.endswith(".png") and "_mesh" not in f and "_qc" not in f
            )
            files = [os.path.join(user_dir, f) for f in files_clean]

        embs: List[np.ndarray] = []
        for fpath in files:
            img = cv2.imread(fpath)
            if img is None or img.size == 0:
                continue
            try:
                emb = recognizer.embed(img)
                embs.append(emb)
            except Exception as e:
                print(f"[GAL] Error embed {fpath}: {e}")

        if embs:
            gallery[user] = np.stack(embs, axis=0)  # (K, D)

    print(f"[GAL] Galería construida: {len(gallery)} usuario(s) para este modelo.")
    return gallery


def get_all_qc_samples(register_root: str) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (usuario, ruta_imagen_qc) para todas las imágenes *_qc.png.
    Si un usuario no tiene _qc, podrías extender esto para usar las normales.
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(register_root):
        return pairs

    for user in sorted(os.listdir(register_root)):
        user_dir = os.path.join(register_root, user)
        if not os.path.isdir(user_dir):
            continue

        for fname in sorted(os.listdir(user_dir)):
            if fname.endswith("_qc.png"):
                img_path = os.path.join(user_dir, fname)
                pairs.append((user, img_path))

    return pairs


def evaluate_model(
    model_name: str,
    recognizer,
    gallery: Dict[str, np.ndarray],
    samples: List[Tuple[str, str]],
) -> None:
    """
    Evalúa un modelo sobre un conjunto de muestras.

    - gallery: {usuario: (K, D) embeddings de registro}
    - samples: lista de (usuario_verdadero, ruta_imagen_qc) a evaluar
    """

    if not gallery:
        print(f"[EVAL:{model_name}] Galería vacía, saltando modelo.")
        return

    if not samples:
        print(f"[EVAL:{model_name}] No hay muestras qc para evaluar.")
        return

    print(f"\n========== EVALUACIÓN MODELO: {model_name} ==========")

    # Métricas de clasificación
    y_true: List[str] = []
    y_pred: List[str] = []

    # Para AUC (genuino vs impostor)
    auc_scores: List[float] = []  # score: -distancia (más alto = más similar)
    auc_labels: List[int] = []    # 1 = genuino, 0 = impostor

    # Medidas de tiempo y recursos
    process = psutil.Process(os.getpid())
    ram_max = process.memory_info().rss  # en bytes

    t0 = time.perf_counter()

    # Umbral por defecto del modelo
    thr = getattr(recognizer, "default_threshold", 0.35)

    for true_user, img_path in samples:
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            print(f"[EVAL:{model_name}] No se pudo leer {img_path}")
            continue

        try:
            emb_q = recognizer.embed(img)
        except Exception as e:
            print(f"[EVAL:{model_name}] Error embed en {img_path}: {e}")
            continue

        # Clasificacion 1:N
        best_user = None
        best_dist = 999.0

        for user, mat in gallery.items():  # mat: (K,D)
            # mínima distancia a cualquiera de las muestras del usuario
            d_min = 999.0
            for ref in mat:
                d = recognizer.cosine_distance(emb_q, ref)
                if d < d_min:
                    d_min = d
            if d_min < best_dist:
                best_dist = d_min
                best_user = user

        # Si pasa el umbral, lo clasificamos como ese usuario; si no, "desconocido"
        if best_user is not None and best_dist <= thr:
            pred_label = best_user
        else:
            pred_label = "desconocido"

        y_true.append(true_user)
        y_pred.append(pred_label)

        # AUC
        for user, mat in gallery.items():
            for ref in mat:
                d = recognizer.cosine_distance(emb_q, ref)
                score = -d  # mayor score → más similar (para ROC)
                label = 1 if user == true_user else 0
                auc_scores.append(score)
                auc_labels.append(label)

        # RAM max
        mem = process.memory_info().rss
        if mem > ram_max:
            ram_max = mem

    total_time = time.perf_counter() - t0
    n_samples = len(y_true)
    fps = n_samples / total_time if total_time > 0 else 0.0

    # CPU aproximado al final del experimento (no perfecto, pero da idea)
    cpu_percent = process.cpu_percent(interval=0.5)
    ram_mb = ram_max / (1024 * 1024)

    # Metricas clasificacion
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    auc_value = None
    if len(set(auc_labels)) == 2:
        auc_value = roc_auc_score(auc_labels, auc_scores)

    print(f"[{model_name}] #muestras evaluadas: {n_samples}")
    print(f"[{model_name}] Tiempo total: {total_time:.3f} s")
    print(f"[{model_name}] FPS efectivos: {fps:.2f} frames/s")
    print(f"[{model_name}] RAM máx: {ram_mb:.1f} MB")
    print(f"[{model_name}] CPU (aprox al final): {cpu_percent:.1f} %")

    print(f"[{model_name}] Accuracy: {acc:.4f}")
    print(f"[{model_name}] Balanced accuracy (macro-like): {bal_acc:.4f}")
    print(f"[{model_name}] Precision macro:   {precision_macro:.4f}")
    print(f"[{model_name}] Recall macro:      {recall_macro:.4f}")
    print(f"[{model_name}] F1-score macro:    {f1_macro:.4f}")
    print(f"[{model_name}] Precision weighted:{precision_weighted:.4f}")
    print(f"[{model_name}] Recall weighted:   {recall_weighted:.4f}")
    print(f"[{model_name}] F1-score weighted: {f1_weighted:.4f}")

    if auc_value is not None:
        print(f"[{model_name}] AUC genuino vs impostor: {auc_value:.4f}")
    else:
        print(f"[{model_name}] AUC: no se pudo calcular (solo una clase en labels).")

    print("\n[Reporte de clasificación por clase]:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    # Cargar qc
    samples = get_all_qc_samples(REGISTER_ROOT)
    if not samples:
        print("No se encontraron imágenes *_qc.png en register. "
              "Asegúrate de haber hecho registros con QC.")
        return

    print(f"Total de muestras QC encontradas: {len(samples)}")

    # Modelos a evaluar
    MODELS = {
        #"buffalo_l": BuffaloLRecognizer(),
        #"facenet_vgg": FacenetVGGFace2Recognizer(),
        "deepface_arcface": DeepFaceRecognizer(model_name="ArcFace"),
        "deepface_sface": DeepFaceRecognizer(model_name="SFace"),
    }

    # Construir galería y evaluar
    for name, rec in MODELS.items():
        print(f"\n==== Preparando modelo '{name}' ====")
        gallery = build_gallery(REGISTER_ROOT, rec)
        evaluate_model(name, rec, gallery, samples)


if __name__ == "__main__":
    main()
