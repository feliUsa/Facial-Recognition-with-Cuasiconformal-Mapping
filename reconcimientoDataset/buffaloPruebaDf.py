import os
import time
import csv
import random
import subprocess
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import psutil
from insightface.app import FaceAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# Ruta donde está el dataset LFW descargado
LFW_ROOT = "/home/daniel/Universidad/experimentosLibrerias/reconocimiento/reconcimientoDataset/modelLFW/4"
IMAGES_ROOT = os.path.join(LFW_ROOT, "lfw-deepfunneled", "lfw-deepfunneled")

# Modelo InsightFace a usar
BUFFALO_MODEL_NAME = "buffalo_l"

# Mínimo de imágenes por persona para incluirla en el experimento
MIN_IMAGES_PER_ID = 5

# Máximo de identidades (None = todas las que cumplan el mínimo)
MAX_IDENTITIES = 100

# Máximo de probes por identidad (además de la imagen de galería)
MAX_PROBES_PER_ID = 5

# Semilla para reproducibilidad
RANDOM_SEED = 42

# Nombre de CSVs
RESULTS_CSV = f"lfw_results_{BUFFALO_MODEL_NAME}.csv"
CONFUSION_CSV = f"lfw_confusion_{BUFFALO_MODEL_NAME}.csv"

# Cada cuántas imágenes muestreamos GPU (nvidia-smi). 0 = no muestrear.
GPU_SAMPLE_EVERY = 0

# Número de negativos por probe para construir ROC/AUC
NEGATIVES_PER_PROBE = 10

# Tamaño máximo de lado de la imagen que mandamos a InsightFace
MAX_IMG_SIDE = 320

# Tamaño de detección de Buffalo (SCRFD)
DET_SIZE = (320, 320)

# Global para el analizador de InsightFace
BUFFALO_APP = None


# =========================
# UTILIDADES GPU
# =========================

def get_gpu_usage():
    """
    Devuelve (memoria_MB, utilizacion_%) del primer GPU usando nvidia-smi.
    Si no hay GPU o falla, devuelve (None, None).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        line = out.decode("utf-8").strip().splitlines()[0]
        mem_str, util_str = [x.strip() for x in line.split(",")]
        mem_mb = float(mem_str)
        mem_util = float(util_str)
        return mem_mb, mem_util
    except Exception:
        return None, None


# =========================
# MÉTRICAS ROC / VERIFICACIÓN
# =========================

def compute_eer_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Calcula el Equal Error Rate (EER) a partir de una curva ROC (fpr, tpr).
    """
    fnr = 1.0 - tpr
    diff = np.abs(fnr - fpr)
    idx = np.argmin(diff)
    eer = 0.5 * (fnr[idx] + fpr[idx])
    return float(eer)


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """
    Devuelve TPR en el punto donde FPR ≈ target_fpr (interpolando).
    """
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    if target_fpr >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(target_fpr, fpr, tpr))


# =========================
# CARGA / PROTOCOLO LFW
# =========================

def build_lfw_protocol(images_root: str,
                       min_images_per_id: int = 5,
                       max_identities: int | None = None,
                       max_probes_per_id: int | None = None):
    """
    Recorre lfw-deepfunneled para construir:
      - una imagen de GALERÍA por persona (imagen de referencia),
      - varias imágenes de PRUEBA (probes) por persona.

    Aplica:
      - min_images_per_id: mínimo de imágenes para que la identidad cuente
      - max_identities:    máximo de identidades
      - max_probes_per_id: máximo de probes por identidad

    Devuelve:
      gallery_paths:   lista de rutas de imágenes de referencia
      gallery_labels:  lista de IDs de persona (strings) de la galería
      probe_paths:     lista de rutas de imágenes de prueba
      probe_labels:    lista de IDs de persona (strings) de las probes
    """
    rng = random.Random(RANDOM_SEED)

    people_dirs = sorted(
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    )

    gallery_paths = []
    gallery_labels = []
    probe_paths = []
    probe_labels = []

    num_ids = 0

    for person in people_dirs:
        person_dir = os.path.join(images_root, person)
        files = [
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith(".jpg")
        ]
        if len(files) < min_images_per_id:
            continue

        rng.shuffle(files)
        ref = files[0]
        probes = files[1:]

        # límite de probes por identidad
        if max_probes_per_id is not None:
            probes = probes[:max_probes_per_id]

        gallery_paths.append(ref)
        gallery_labels.append(person)

        probe_paths.extend(probes)
        probe_labels.extend([person] * len(probes))

        num_ids += 1
        if max_identities is not None and num_ids >= max_identities:
            break

    return gallery_paths, gallery_labels, probe_paths, probe_labels


def embed_image(img_path: str):
    """
    Obtiene el embedding de Buffalo (buffalo_l) para una imagen usando InsightFace.

    - Usa FaceAnalysis(name=BUFFALO_MODEL_NAME) ya inicializado en BUFFALO_APP.
    - Reescala la imagen para no pasar tamaños gigantes.
    - Devuelve un vector np.float32 (ya L2-normalizado en normed_embedding) o None si no se detecta cara.
    """
    global BUFFALO_APP
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Reducir tamaño máximo de lado para acelerar detección
    if MAX_IMG_SIDE > 0:
        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side > MAX_IMG_SIDE:
            scale = MAX_IMG_SIDE / float(max_side)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    faces = BUFFALO_APP.get(img)
    if not faces:
        return None

    # Nos quedamos con la cara con mayor score de detección
    face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))

    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return None

    emb = np.array(emb, dtype=np.float32)
    return emb


# =========================
# EXPERIMENTO PRINCIPAL
# =========================

def run_buffalo_lfw():
    global BUFFALO_APP

    print("===============================================")
    print(f"[INFO] Experimento LFW con InsightFace ({BUFFALO_MODEL_NAME})")
    print("===============================================")
    print(f"[INFO] IMAGES_ROOT = {IMAGES_ROOT}")

    # 1) Construir protocolo (galería/probes) a partir de LFW
    gallery_paths, gallery_labels, probe_paths, probe_labels = build_lfw_protocol(
        IMAGES_ROOT,
        min_images_per_id=MIN_IMAGES_PER_ID,
        max_identities=MAX_IDENTITIES,
        max_probes_per_id=MAX_PROBES_PER_ID,
    )

    print(f"[INFO] Ids encontradas en protocolo:           {len(set(gallery_labels))}")
    print(f"[INFO] Imágenes galería (raw):                {len(gallery_paths)}")
    print(f"[INFO] Imágenes probes (raw, ya limitadas):   {len(probe_paths)}")

    if len(gallery_paths) == 0 or len(probe_paths) == 0:
        print("[ERROR] Muy pocas imágenes después de aplicar MIN_IMAGES_PER_ID / MAX_IDENTITIES / MAX_PROBES_PER_ID.")
        return

    # 2) Inicializar InsightFace (Buffalo)
    print(f"\n[INFO] Inicializando InsightFace ({BUFFALO_MODEL_NAME})...")
    BUFFALO_APP = FaceAnalysis(name=BUFFALO_MODEL_NAME)
    # ctx_id=-1 -> CPU; det_size reducido para acelerar
    BUFFALO_APP.prepare(ctx_id=-1, det_size=DET_SIZE)
    print("[INFO] InsightFace preparado.")

    # 3) Métricas de recursos
    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # primer muestreo

    cpu_samples = []
    ram_samples = []
    gpu_mem_samples = []
    gpu_util_samples = []

    total_embed_time = 0.0
    total_classify_time = 0.0

    # ============
    # Embeddings GALERÍA
    # ============
    print("\n[INFO] Calculando embeddings de GALERÍA...")
    gallery_embs = []
    valid_gallery_paths = []
    valid_gallery_labels = []
    failed_gallery = 0

    t0_gallery = time.time()
    for idx, path in enumerate(gallery_paths):
        t_start = time.time()
        emb = embed_image(path)
        t_end = time.time()
        total_embed_time += (t_end - t_start)

        if emb is None:
            failed_gallery += 1
        else:
            gallery_embs.append(emb)
            valid_gallery_paths.append(path)
            valid_gallery_labels.append(gallery_labels[idx])

        # Recursos
        cpu_pct = process.cpu_percent(interval=None)
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_samples.append(cpu_pct)
        ram_samples.append(mem_mb)

        if GPU_SAMPLE_EVERY > 0 and (idx % GPU_SAMPLE_EVERY == 0):
            gm, gu = get_gpu_usage()
            if gm is not None:
                gpu_mem_samples.append(gm)
                gpu_util_samples.append(gu)

        if (idx + 1) % 20 == 0:
            print(f"  - {idx+1}/{len(gallery_paths)} imágenes galería procesadas...")

    t1_gallery = time.time()
    print(f"[INFO] Tiempo galería: {t1_gallery - t0_gallery:.2f} s")
    print(f"[INFO] Imágenes de galería fallidas (sin cara): {failed_gallery}")

    if len(gallery_embs) == 0:
        print("[ERROR] No se pudo generar ningún embedding para la galería.")
        return

    gallery_embs = np.stack(gallery_embs, axis=0)
    gallery_paths = valid_gallery_paths
    gallery_labels = valid_gallery_labels
    n_gallery = len(gallery_paths)
    ids_with_gallery = set(gallery_labels)

    # ============
    # Filtrar probes para que solo haya ids presentes en la galería válida
    # ============
    filtered_probe_paths = []
    filtered_probe_labels = []
    for p, lab in zip(probe_paths, probe_labels):
        if lab in ids_with_gallery:
            filtered_probe_paths.append(p)
            filtered_probe_labels.append(lab)

    probe_paths = filtered_probe_paths
    probe_labels = filtered_probe_labels

    if len(probe_paths) == 0:
        print("[ERROR] Después de filtrar por identidades válidas, no quedan probes.")
        return

    # ============
    # Embeddings PROBES
    # ============
    print("\n[INFO] Calculando embeddings de PROBES...")
    probe_embs = []
    valid_probe_paths = []
    valid_probe_labels = []
    failed_probes = 0

    t0_probe = time.time()
    for idx, path in enumerate(probe_paths):
        t_start = time.time()
        emb = embed_image(path)
        t_end = time.time()
        total_embed_time += (t_end - t_start)

        if emb is None:
            failed_probes += 1
        else:
            probe_embs.append(emb)
            valid_probe_paths.append(path)
            valid_probe_labels.append(probe_labels[idx])

        cpu_pct = process.cpu_percent(interval=None)
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_samples.append(cpu_pct)
        ram_samples.append(mem_mb)

        if GPU_SAMPLE_EVERY > 0 and (idx % GPU_SAMPLE_EVERY == 0):
            gm, gu = get_gpu_usage()
            if gm is not None:
                gpu_mem_samples.append(gm)
                gpu_util_samples.append(gu)

        if (idx + 1) % 50 == 0:
            print(f"  - {idx+1}/{len(probe_paths)} probes procesadas...")

    t1_probe = time.time()
    print(f"[INFO] Tiempo probes: {t1_probe - t0_probe:.2f} s")
    print(f"[INFO] Probes fallidas (sin cara): {failed_probes}")

    if len(probe_embs) == 0:
        print("[ERROR] No se pudo generar ningún embedding para las probes.")
        return

    probe_embs = np.stack(probe_embs, axis=0)
    probe_paths = valid_probe_paths
    probe_labels = valid_probe_labels

    n_probes = len(probe_paths)
    unique_ids = sorted(set(gallery_labels))  # identidades realmente presentes
    print(f"[INFO] Identidades usadas (final):     {len(unique_ids)}")
    print(f"[INFO] Imágenes en galería (válidas):  {n_gallery}")
    print(f"[INFO] Imágenes de prueba (válidas):   {n_probes}")

    # Mapear label -> índice en galería
    label_to_gallery_idx = {lab: i for i, lab in enumerate(gallery_labels)}

    # ============
    # 4) Clasificación 1-NN en espacio de embeddings (cosine = dot)
    # ============
    print("\n[INFO] Calculando similitudes y predicciones...")
    t0_class = time.time()
    S = probe_embs @ gallery_embs.T  # (n_probes, n_gallery)
    t1_class = time.time()
    total_classify_time += (t1_class - t0_class)

    best_idx = np.argmax(S, axis=1)
    pred_labels = [gallery_labels[i] for i in best_idx]
    true_labels = probe_labels

    # ============
    # 5) ROC / AUC / EER / TPR@FPR
    # ============
    print("[INFO] Construyendo scores para ROC/AUC...")
    rng = random.Random(RANDOM_SEED + 1)
    pos_scores = []
    neg_scores = []

    all_gallery_indices = list(range(n_gallery))

    for i in range(n_probes):
        true_lab = true_labels[i]
        gi = label_to_gallery_idx[true_lab]
        sim_row = S[i]

        # score positivo
        pos_scores.append(sim_row[gi])

        # scores negativos
        neg_candidates = [idx for idx in all_gallery_indices if idx != gi]
        k = min(NEGATIVES_PER_PROBE, len(neg_candidates))
        if k > 0:
            neg_sample = rng.sample(neg_candidates, k)
            for j in neg_sample:
                neg_scores.append(sim_row[j])

    y_true = np.concatenate([
        np.ones(len(pos_scores), dtype=np.int32),
        np.zeros(len(neg_scores), dtype=np.int32),
    ])
    scores_all = np.concatenate([
        np.array(pos_scores, dtype=np.float32),
        np.array(neg_scores, dtype=np.float32),
    ])

    try:
        auc = roc_auc_score(y_true, scores_all)
        fpr, tpr, _ = roc_curve(y_true, scores_all)
        eer = compute_eer_from_roc(fpr, tpr)
        tpr_1e2 = tpr_at_fpr(fpr, tpr, 1e-2)
        tpr_1e3 = tpr_at_fpr(fpr, tpr, 1e-3)
        tpr_1e4 = tpr_at_fpr(fpr, tpr, 1e-4)
    except Exception:
        auc = float("nan")
        eer = float("nan")
        tpr_1e2 = float("nan")
        tpr_1e3 = float("nan")
        tpr_1e4 = float("nan")

    # ============
    # 6) Métricas multi-clase
    # ============
    print("[INFO] Calculando métricas globales (clasificación)...")
    acc = accuracy_score(true_labels, pred_labels)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    labels_sorted = sorted(unique_ids)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_sorted)

    # ============
    # 7) Métricas de tiempo / recursos
    # ============
    num_embeddings_done = gallery_embs.shape[0] + probe_embs.shape[0]
    avg_embed_time_ms = (total_embed_time / num_embeddings_done) * 1000.0
    fps_embed = num_embeddings_done / total_embed_time if total_embed_time > 0 else 0.0
    avg_class_time_ms = (total_classify_time / n_probes) * 1000.0

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0.0
    avg_gpu_mem = sum(gpu_mem_samples) / len(gpu_mem_samples) if gpu_mem_samples else None
    avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples) if gpu_util_samples else None

    # ============
    # 8) Guardar CSV por imagen (probes)
    # ============
    print(f"[INFO] Guardando resultados por imagen en {RESULTS_CSV} ...")
    with open(RESULTS_CSV, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "probe_path",
            "true_label",
            "pred_label",
            "correct",
            "similarity_best",
            "similarity_true_gallery",
        ])
        for i in range(n_probes):
            true_lab = true_labels[i]
            pred_lab = pred_labels[i]
            correct = int(true_lab == pred_lab)

            gi_true = label_to_gallery_idx[true_lab]
            sim_row = S[i]
            sim_best = sim_row[best_idx[i]]
            sim_true = sim_row[gi_true]

            writer.writerow([
                probe_paths[i],
                true_lab,
                pred_lab,
                correct,
                f"{sim_best:.6f}",
                f"{sim_true:.6f}",
            ])

    # ============
    # 9) Guardar matriz de confusión
    # ============
    print(f"[INFO] Guardando matriz de confusión en {CONFUSION_CSV} ...")
    with open(CONFUSION_CSV, "w", newline="") as f_cm:
        writer = csv.writer(f_cm)
        writer.writerow(["label"] + labels_sorted)
        for i, lab in enumerate(labels_sorted):
            row = [lab] + list(cm[i])
            writer.writerow(row)

    # ============
    # 10) Resumen
    # ============
    print(f"\n========== RESULTADOS GLOBALES (InsightFace {BUFFALO_MODEL_NAME} + LFW) ==========")
    print(f"Identidades usadas:           {len(unique_ids)}")
    print(f"Imágenes galería válidas:     {n_gallery}")
    print(f"Imágenes probe válidas:       {n_probes}")
    print(f"Galería sin cara:             {failed_gallery}")
    print(f"Probes sin cara:              {failed_probes}")
    print("")
    print(f"Accuracy:                     {acc:.4f}")
    print(f"Precision macro:              {prec_macro:.4f}")
    print(f"Recall macro:                 {rec_macro:.4f}")
    print(f"F1-score macro:               {f1_macro:.4f}")
    print(f"Precision weighted:           {prec_weighted:.4f}")
    print(f"Recall weighted:              {rec_weighted:.4f}")
    print(f"F1-score weighted:            {f1_weighted:.4f}")
    print("")
    print(f"AUC (ROC, verificación):      {auc:.4f}")
    print(f"EER:                          {eer:.4f}")
    print(f"TPR @ FPR=1e-2:               {tpr_1e2:.4f}")
    print(f"TPR @ FPR=1e-3:               {tpr_1e3:.4f}")
    print(f"TPR @ FPR=1e-4:               {tpr_1e4:.4f}")
    print("")
    print(f"Tiempo total embedding:       {total_embed_time:.2f} s")
    print(f"Tiempo medio embedding/img:   {avg_embed_time_ms:.2f} ms")
    print(f"FPS aproximado embedding:     {fps_embed:.2f} img/s")
    print(f"Tiempo medio clasificación:   {avg_class_time_ms:.4f} ms/probe")
    print("")
    print(f"CPU medio proceso:            {avg_cpu:.2f} %")
    print(f"RAM media proceso:            {avg_ram:.2f} MB")
    if avg_gpu_mem is not None:
        print(f"GPU memoria media:            {avg_gpu_mem:.2f} MB")
        print(f"GPU uso medio:                {avg_gpu_util:.2f} %")
    else:
        print("GPU: no se pudieron obtener métricas (probablemente no hay nvidia-smi).")
    print("")
    print(f"Resultados por imagen guardados en: {RESULTS_CSV}")
    print(f"Matriz de confusión guardada en:    {CONFUSION_CSV}")
    print("==================================================================")


if __name__ == "__main__":
    run_buffalo_lfw()
