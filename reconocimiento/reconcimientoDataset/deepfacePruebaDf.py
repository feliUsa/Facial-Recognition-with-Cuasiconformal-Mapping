import os
import time
import csv
import random
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import psutil
from deepface import DeepFace
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)



LFW_ROOT = "/home/daniel/Universidad/experimentosLibrerias/reconocimiento/reconcimientoDataset/modelLFW/4"
IMAGES_ROOT = os.path.join(LFW_ROOT, "lfw-deepfunneled", "lfw-deepfunneled")

# "ArcFace" o "SFace"
MODEL_NAME = "SFace"

# Minimo de imágenes por persona para incluirla en el experimento
MIN_IMAGES_PER_ID = 5

# Máximo de identidades para no morir en tiempo (None = todas las que cumplan mínimo)
MAX_IDENTITIES = 300

# Semilla para reproducibilidad
RANDOM_SEED = 42

# Nombre de CSV de resultados por imagen y de matriz de confusión
RESULTS_CSV = f"lfw_results_{MODEL_NAME}.csv"
CONFUSION_CSV = f"lfw_confusion_{MODEL_NAME}.csv"

# Cada cuántas imágenes muestreamos GPU (nvidia-smi). 0 = no muestrear.
GPU_SAMPLE_EVERY = 20

# Número de negativos por probe para construir ROC/AUC
# (subido un poco para que el AUC sea más estable)
NEGATIVES_PER_PROBE = 50




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
        util_pct = float(util_str)
        return mem_mb, util_pct
    except Exception:
        return None, None



def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    """
    Calcula el Equal Error Rate (EER) a partir de una curva ROC.

    EER es el punto donde FPR ~= FNR (=1-TPR).
    Devolvemos:
      - eer (float)
      - thr_eer (umbral en el espacio de scores)
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    thresholds = np.asarray(thresholds)

    fnr = 1.0 - tpr
    abs_diff = np.abs(fnr - fpr)
    idx = np.nanargmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr)


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float):
    """
    Devuelve el TPR correspondiente al primer punto de la ROC
    donde FPR >= target_fpr. Si la ROC no llega a ese FPR,
    devolvemos el último TPR disponible.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    idxs = np.where(fpr >= target_fpr)[0]
    if len(idxs) == 0:
        # No se alcanza ese FPR, devolvemos el último TPR
        return float(tpr[-1])
    return float(tpr[idxs[0]])


# Protocolo LWF

def build_lfw_protocol(images_root: str,
                       min_images_per_id: int = 5,
                       max_identities: int | None = None):
    """
    Recorre lfw-deepfunneled para construir:
      - una imagen de GALERÍA por persona (imagen de referencia),
      - varias imágenes de PRUEBA (probes) por persona.

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

        gallery_paths.append(ref)
        gallery_labels.append(person)

        probe_paths.extend(probes)
        probe_labels.extend([person] * len(probes))

        num_ids += 1
        if max_identities is not None and num_ids >= max_identities:
            break

    return gallery_paths, gallery_labels, probe_paths, probe_labels


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def embed_image(img_path: str, model_name: str):
    """
    Obtiene el embedding DeepFace de una imagen.

    Usamos:
      - detector_backend="skip" y enforce_detection=False
        porque LFW ya está bastante recortado/alineado y queremos velocidad.
    """
    rep = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    # DeepFace.represent devuelve lista de dicts
    emb = np.array(rep[0]["embedding"], dtype=np.float32)
    return l2_normalize(emb)



def run_deepface_lfw():
    print("===============================================")
    print(f"Experimento LFW con DeepFace ({MODEL_NAME})")
    print("===============================================")
    print(f"IMAGES_ROOT = {IMAGES_ROOT}")

    # Construir protocolo (galeria)
    gallery_paths, gallery_labels, probe_paths, probe_labels = build_lfw_protocol(
        IMAGES_ROOT,
        min_images_per_id=MIN_IMAGES_PER_ID,
        max_identities=MAX_IDENTITIES,
    )

    n_gallery = len(gallery_paths)
    n_probes = len(probe_paths)
    unique_ids = sorted(set(gallery_labels))
    assert len(unique_ids) == n_gallery, "Cada persona debería tener solo una imagen de galería."

    print(f"Identidades usadas:           {len(unique_ids)}")
    print(f"Imágenes en galería:         {n_gallery}")
    print(f"Imágenes de prueba (probes): {n_probes}")

    if n_gallery == 0 or n_probes == 0:
        print("Muy pocas imágenes después de aplicar MIN_IMAGES_PER_ID / MAX_IDENTITIES.")
        return

    # Mapear label -> índice en galería
    label_to_gallery_idx = {lab: i for i, lab in enumerate(gallery_labels)}

    # Embeddings
    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # primer muestreo

    cpu_samples = []
    ram_samples = []
    gpu_mem_samples = []
    gpu_util_samples = []

    # Tiempo de embedding total
    total_embed_time = 0.0
    total_classify_time = 0.0

    # Embeddings de la galeria
    print("\n[INFO] Calculando embeddings de GALERÍA...")
    gallery_embs = []
    t0_gallery = time.time()
    for idx, path in enumerate(gallery_paths):
        t_start = time.time()
        emb = embed_image(path, MODEL_NAME)
        t_end = time.time()
        total_embed_time += (t_end - t_start)
        gallery_embs.append(emb)

        # Métricas CPU/RAM
        cpu_pct = process.cpu_percent(interval=None)
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_samples.append(cpu_pct)
        ram_samples.append(mem_mb)

        # GPU (muestra cada GPU_SAMPLE_EVERY)
        gpu_mem_mb = None
        gpu_util_pct = None
        if GPU_SAMPLE_EVERY > 0 and (idx % GPU_SAMPLE_EVERY == 0):
            gm, gu = get_gpu_usage()
            if gm is not None:
                gpu_mem_mb = gm
                gpu_util_pct = gu
                gpu_mem_samples.append(gm)
                gpu_util_samples.append(gu)

        if (idx + 1) % 50 == 0:
            print(f"  - {idx+1}/{n_gallery} embeddings de galería listos...")
    t1_gallery = time.time()
    print(f"[INFO] Tiempo galería: {t1_gallery - t0_gallery:.2f} s")

    gallery_embs = np.stack(gallery_embs, axis=0)  # (n_gallery, d)

    # Embeddings de las probes
    print("\n[INFO] Calculando embeddings de PROBES...")
    probe_embs = []
    t0_probe = time.time()
    for idx, path in enumerate(probe_paths):
        t_start = time.time()
        emb = embed_image(path, MODEL_NAME)
        t_end = time.time()
        total_embed_time += (t_end - t_start)
        probe_embs.append(emb)

        cpu_pct = process.cpu_percent(interval=None)
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_samples.append(cpu_pct)
        ram_samples.append(mem_mb)

        gpu_mem_mb = None
        gpu_util_pct = None
        if GPU_SAMPLE_EVERY > 0 and (idx % GPU_SAMPLE_EVERY == 0):
            gm, gu = get_gpu_usage()
            if gm is not None:
                gpu_mem_mb = gm
                gpu_util_pct = gu
                gpu_mem_samples.append(gm)
                gpu_util_samples.append(gu)

        if (idx + 1) % 200 == 0:
            print(f"  - {idx+1}/{n_probes} embeddings de probes listos...")
    t1_probe = time.time()
    print(f"[INFO] Tiempo probes: {t1_probe - t0_probe:.2f} s")

    probe_embs = np.stack(probe_embs, axis=0)  # (n_probes, d)

    # Clasificación 1-N en espacio de embeddings (cosine similarity)
    print("\n[INFO] Calculando similitudes y predicciones...")
    t0_class = time.time()
    # Como están normalizados, la similitud coseno es simplemente el producto punto
    # S: (n_probes, n_gallery)
    S = probe_embs @ gallery_embs.T
    t1_class = time.time()
    total_classify_time += (t1_class - t0_class)

    # Predicción = galería con mayor similitud
    best_idx = np.argmax(S, axis=1)             # (n_probes,)
    pred_labels = [gallery_labels[i] for i in best_idx]
    true_labels = probe_labels

    # ROC / AUC / EER / TPR@FPR
    print("Construyendo scores para ROC/AUC/EER...")
    rng = random.Random(RANDOM_SEED + 1)
    pos_scores = []
    neg_scores = []

    all_gallery_indices = list(range(n_gallery))

    for i in range(n_probes):
        true_lab = true_labels[i]
        gi = label_to_gallery_idx[true_lab]
        sim_row = S[i]

        # score positivo: similitud con su galería verdadera
        pos_scores.append(sim_row[gi])

        # scores negativos: similitudes con algunas galerías de otras identidades
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

    # AUC
    try:
        auc = roc_auc_score(y_true, scores_all)
    except Exception:
        auc = float("nan")

    # Curva ROC completa (para EER y TPR@FPR)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, scores_all)
        eer, eer_thr = compute_eer(fpr, tpr, thresholds)
        tpr_fpr_1e2 = tpr_at_fpr(fpr, tpr, 1e-2)   # TPR @ FPR=0.01
        tpr_fpr_1e3 = tpr_at_fpr(fpr, tpr, 1e-3)   # TPR @ FPR=0.001
    except Exception:
        fpr = tpr = thresholds = None
        eer = eer_thr = float("nan")
        tpr_fpr_1e2 = tpr_fpr_1e3 = float("nan")

    # Metricas de clasificación multi-clase
    print("Calculando métricas globales...")
    acc = accuracy_score(true_labels, pred_labels)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    # Confusion matrix
    labels_sorted = sorted(unique_ids)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_sorted)

    # Metricas de tiempo / recursos
    total_images = n_gallery + n_probes
    avg_embed_time_ms = (total_embed_time / total_images) * 1000.0
    fps_embed = total_images / total_embed_time if total_embed_time > 0 else 0.0
    avg_class_time_ms = (total_classify_time / n_probes) * 1000.0

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0.0
    avg_gpu_mem = sum(gpu_mem_samples) / len(gpu_mem_samples) if gpu_mem_samples else None
    avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples) if gpu_util_samples else None

    # Guardar CSV
    print(f"Guardando resultados por imagen en {RESULTS_CSV} ...")
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

    # Guardar matriz de confusión
    print(f"[INFO] Guardando matriz de confusión en {CONFUSION_CSV} ...")
    with open(CONFUSION_CSV, "w", newline="") as f_cm:
        writer = csv.writer(f_cm)
        # primera fila: encabezados ("" + labels)
        writer.writerow(["label"] + labels_sorted)
        for i, lab in enumerate(labels_sorted):
            row = [lab] + list(cm[i])
            writer.writerow(row)

    print("\n========== RESULTADOS GLOBALES (DeepFace + LFW) ==========")
    print(f"Modelo DeepFace:              {MODEL_NAME}")
    print(f"Identidades usadas:           {len(unique_ids)}")
    print(f"Imágenes galería:             {n_gallery}")
    print(f"Imágenes prueba (probes):     {n_probes}")
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
    print(f"EER:                          {eer:.4f}  (umbral = {eer_thr:.4f})")
    print(f"TPR @ FPR=1e-2:               {tpr_fpr_1e2:.4f}")
    print(f"TPR @ FPR=1e-3:               {tpr_fpr_1e3:.4f}")
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
        print(f"GPU uso medio:               {avg_gpu_util:.2f} %")
    else:
        print("GPU: no se pudieron obtener métricas (probablemente no hay nvidia-smi).")
    print("")
    print(f"Resultados por imagen guardados en: {RESULTS_CSV}")
    print(f"Matriz de confusión guardada en:    {CONFUSION_CSV}")
    print("==========================================================")


if __name__ == "__main__":
    run_deepface_lfw()
