import os
import time
import csv
import random
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization


LFW_ROOT = "/home/daniel/Universidad/experimentosLibrerias/reconocimiento/reconcimientoDataset/modelLFW/4"
IMAGES_ROOT = os.path.join(LFW_ROOT, "lfw-deepfunneled", "lfw-deepfunneled")


FACENET_WEIGHTS = "vggface2"
MODEL_TAG = f"facenet_{FACENET_WEIGHTS}"

# Minimo imagenes por persona
MIN_IMAGES_PER_ID = 5

# Maximo de identidades (None = todas las que cumplan el mínimo)
MAX_IDENTITIES = 300

# Maximo de probes por identidad (para controlar tiempo)
MAX_PROBES_PER_ID = 5

# Semilla para reproducibilidad
RANDOM_SEED = 42

RESULTS_CSV = f"lfw_results_{MODEL_TAG}.csv"
CONFUSION_CSV = f"lfw_confusion_{MODEL_TAG}.csv"

# Cada cuántas imágenes muestreamos GPU (nvidia-smi). 0 = no muestrear.
GPU_SAMPLE_EVERY = 20

# Número de negativos por probe para construir ROC/AUC
NEGATIVES_PER_PROBE = 10

# Tamaño de la imagen de entrada para InceptionResnetV1
FACENET_IMG_SIZE = 160

# Dispositivo (CPU / GPU si disponible)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Modelo global
FACENET_MODEL: Optional[InceptionResnetV1] = None



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



def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Calcula el Equal Error Rate (EER) a partir de etiquetas binarias y scores.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
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


def build_lfw_protocol(
    images_root: str,
    min_images_per_id: int = 5,
    max_identities: Optional[int] = None,
    max_probes_per_id: Optional[int] = None,
):
    """
    Recorre lfw-deepfunneled para construir:
      - una imagen de GALERÍA por persona (imagen de referencia),
      - varias imágenes de PRUEBA (probes) por persona (limitadas por max_probes_per_id).

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

        # Limitar número de probes por identidad si se define
        if max_probes_per_id is not None:
            probes = probes[:max_probes_per_id]

        if len(probes) == 0:
            continue

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


def embed_image(img_path: str) -> Optional[np.ndarray]:
    """
    Obtiene el embedding FaceNet (facenet-pytorch) para una imagen:

      - Carga la imagen con PIL.
      - Redimensiona a 160x160.
      - Convierte a tensor y aplica fixed_image_standardization.
      - Pasa por InceptionResnetV1 preentrenado.
      - Devuelve un vector L2-normalizado (np.float32).

    Si hay error al leer la imagen, devuelve None.
    """
    global FACENET_MODEL

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    img = img.resize((FACENET_IMG_SIZE, FACENET_IMG_SIZE))

    # Convertir a tensor (C,H,W) con rango [0,255] float32
    img_np = np.asarray(img).astype(np.float32)  # H,W,3
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # 3,H,W

    # Normalización fija tipo FaceNet
    img_tensor = fixed_image_standardization(img_tensor)

    # Añadir batch y mover a dispositivo
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = FACENET_MODEL(img_tensor).cpu().numpy()[0]

    emb = emb.astype(np.float32)
    emb = l2_normalize(emb)
    return emb


def run_facenet_lfw():
    global FACENET_MODEL

    print("===============================================")
    print(f"[INFO] Experimento LFW con facenet-pytorch ({FACENET_WEIGHTS})")
    print("===============================================")
    print(f"[INFO] IMAGES_ROOT = {IMAGES_ROOT}")
    print(f"[INFO] DEVICE      = {DEVICE}")

    # Construir protocolo (galería/probes) a partir de LFW
    gallery_paths, gallery_labels, probe_paths, probe_labels = build_lfw_protocol(
        IMAGES_ROOT,
        min_images_per_id=MIN_IMAGES_PER_ID,
        max_identities=MAX_IDENTITIES,
        max_probes_per_id=MAX_PROBES_PER_ID,
    )

    print(f"[INFO] Identidades usadas (raw):      {len(set(gallery_labels))}")
    print(f"[INFO] Imagenes galería (raw):        {len(gallery_paths)}")
    print(f"[INFO] Imagenes probes (raw):         {len(probe_paths)}")

    if len(gallery_paths) == 0 or len(probe_paths) == 0:
        print("[ERROR] Muy pocas imágenes después de aplicar MIN_IMAGES_PER_ID / MAX_IDENTITIES.")
        return

    # Inicializar modelo FaceNet
    print("\n[INFO] Inicializando InceptionResnetV1 (facenet-pytorch)...")
    FACENET_MODEL = InceptionResnetV1(pretrained=FACENET_WEIGHTS).eval().to(DEVICE)
    for p in FACENET_MODEL.parameters():
        p.requires_grad = False
    print("[INFO] Modelo FaceNet preparado.")

    # Mtricas de recursos
    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # primer muestreo

    cpu_samples = []
    ram_samples = []
    gpu_mem_samples = []
    gpu_util_samples = []

    total_embed_time = 0.0
    total_classify_time = 0.0


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

        if (idx + 1) % 50 == 0:
            print(f"  - {idx+1}/{len(gallery_paths)} imágenes galería procesadas...")

    t1_gallery = time.time()
    print(f"[INFO] Tiempo galería: {t1_gallery - t0_gallery:.2f} s")
    print(f"[INFO] Imágenes de galería fallidas (no emb): {failed_gallery}")

    if len(gallery_embs) == 0:
        print("[ERROR] No se pudo generar ningún embedding para la galería.")
        return

    gallery_embs = np.stack(gallery_embs, axis=0)
    gallery_paths = valid_gallery_paths
    gallery_labels = valid_gallery_labels
    n_gallery = len(gallery_paths)
    ids_with_gallery = set(gallery_labels)


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

        if (idx + 1) % 200 == 0:
            print(f"  - {idx+1}/{len(probe_paths)} probes procesadas...")

    t1_probe = time.time()
    print(f"[INFO] Tiempo probes: {t1_probe - t0_probe:.2f} s")
    print(f"[INFO] Probes fallidas (no emb): {failed_probes}")

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

    # Clasificacion
    print("\n[INFO] Calculando similitudes y predicciones...")
    t0_class = time.time()
    S = probe_embs @ gallery_embs.T  # (n_probes, n_gallery)
    t1_class = time.time()
    total_classify_time += (t1_class - t0_class)

    best_idx = np.argmax(S, axis=1)
    pred_labels = [gallery_labels[i] for i in best_idx]
    true_labels = probe_labels

    # 5) ROC / AUC / EER / TPR y FPR
    print("[INFO] Construyendo scores para ROC/AUC...")
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

    try:
        auc = roc_auc_score(y_true, scores_all)
        fpr, tpr, _ = roc_curve(y_true, scores_all)
        eer = compute_eer(y_true, scores_all)
        tpr_1e2 = tpr_at_fpr(fpr, tpr, 1e-2)
        tpr_1e3 = tpr_at_fpr(fpr, tpr, 1e-3)
        tpr_1e4 = tpr_at_fpr(fpr, tpr, 1e-4)
    except Exception:
        auc = float("nan")
        eer = float("nan")
        tpr_1e2 = float("nan")
        tpr_1e3 = float("nan")
        tpr_1e4 = float("nan")

    # METRICAS
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


    num_embeddings_done = gallery_embs.shape[0] + probe_embs.shape[0]
    avg_embed_time_ms = (total_embed_time / num_embeddings_done) * 1000.0
    fps_embed = num_embeddings_done / total_embed_time if total_embed_time > 0 else 0.0
    avg_class_time_ms = (total_classify_time / n_probes) * 1000.0

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0.0
    avg_gpu_mem = sum(gpu_mem_samples) / len(gpu_mem_samples) if gpu_mem_samples else None
    avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples) if gpu_util_samples else None


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


    print(f"[INFO] Guardando matriz de confusión en {CONFUSION_CSV} ...")
    with open(CONFUSION_CSV, "w", newline="") as f_cm:
        writer = csv.writer(f_cm)
        writer.writerow(["label"] + labels_sorted)
        for i, lab in enumerate(labels_sorted):
            row = [lab] + list(cm[i])
            writer.writerow(row)

    print("\n========== RESULTADOS GLOBALES (facenet-pytorch + LFW) ==========")
    print(f"Modelo FaceNet (facenet-pytorch): {FACENET_WEIGHTS}")
    print(f"Identidades usadas:               {len(unique_ids)}")
    print(f"Imágenes galería válidas:         {n_gallery}")
    print(f"Imágenes probe válidas:           {n_probes}")
    print(f"Galería sin embedding:            {failed_gallery}")
    print(f"Probes sin embedding:             {failed_probes}")
    print("")
    print(f"Accuracy:                         {acc:.4f}")
    print(f"Precision macro:                  {prec_macro:.4f}")
    print(f"Recall macro:                     {rec_macro:.4f}")
    print(f"F1-score macro:                   {f1_macro:.4f}")
    print(f"Precision weighted:               {prec_weighted:.4f}")
    print(f"Recall weighted:                  {rec_weighted:.4f}")
    print(f"F1-score weighted:                {f1_weighted:.4f}")
    print("")
    print(f"AUC (ROC, verificación):          {auc:.4f}")
    print(f"EER:                              {eer:.4f}")
    print(f"TPR @ FPR=1e-2:                   {tpr_1e2:.4f}")
    print(f"TPR @ FPR=1e-3:                   {tpr_1e3:.4f}")
    print(f"TPR @ FPR=1e-4:                   {tpr_1e4:.4f}")
    print("")
    print(f"Tiempo total embedding:           {total_embed_time:.2f} s")
    print(f"Tiempo medio embedding/img:       {avg_embed_time_ms:.2f} ms")
    print(f"FPS aproximado embedding:         {fps_embed:.2f} img/s")
    print(f"Tiempo medio clasificación:       {avg_class_time_ms:.4f} ms/probe")
    print("")
    print(f"CPU medio proceso:                {avg_cpu:.2f} %")
    print(f"RAM media proceso:                {avg_ram:.2f} MB")
    if avg_gpu_mem is not None:
        print(f"GPU memoria media:                {avg_gpu_mem:.2f} MB")
        print(f"GPU uso medio:                    {avg_gpu_util:.2f} %")
    else:
        print("GPU: no se pudieron obtener métricas (probablemente no hay nvidia-smi).")
    print("")
    print(f"Resultados por imagen guardados en: {RESULTS_CSV}")
    print(f"Matriz de confusión guardada en:    {CONFUSION_CSV}")
    print("==================================================================")


if __name__ == "__main__":
    run_facenet_lfw()
