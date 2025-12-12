import os
import time
import csv
import subprocess
from pathlib import Path
import random

import cv2
import numpy as np
import psutil



ROOT_DIR = "/home/daniel/Universidad/experimentosLibrerias/identificacion/modelos/5"

# Ruta al modelo YuNet ONNX
MODEL_PATH = "/home/daniel/Universidad/experimentosLibrerias/identificacion/modelos/face_detection_yunet_2023mar.onnx"

SPLIT = "train"          # "train" o "val"
IOU_THRESHOLD = 0.5      # umbral de IoU para contar TP
MAX_IMAGES = 5000        # None para usar todas las imágenes
RESULTS_CSV = "yunet_wider_results.csv"

# Visualización (ejemplos aleatorios)
VISUAL_DEBUG = True
MAX_VISUAL_EXAMPLES = 5      # máximo de imágenes que se mostrarán
VISUAL_SAMPLE_PROB = 0.01    # probabilidad de guardar un ejemplo en cada imagen

# Cada cuántas imágenes muestreamos GPU (nvidia-smi). 0 = no muestrear.
GPU_SAMPLE_EVERY = 20



def load_wider_annotations(root_dir: Path, split: str = "train"):
    """
    Carga anotaciones de WIDER FACE con la estructura:

    root_dir/
      WIDER_train/
      WIDER_val/
      wider_face_train_bbx_gt.txt
      wider_face_val_bbx_gt.txt

    y además soporta el caso:
      root_dir/WIDER_train/WIDER_train/images/...
    """

    ann_file = root_dir / f"wider_face_{split}_bbx_gt.txt"

    # Determinar dónde están realmente las imágenes
    cand1 = root_dir / f"WIDER_{split}" / "images"
    cand2 = root_dir / f"WIDER_{split}"
    cand3 = root_dir / f"WIDER_{split}" / f"WIDER_{split}" / "images"

    if cand1.exists():
        img_root = cand1
    elif cand3.exists():
        img_root = cand3 
    elif cand2.exists():
        img_root = cand2
    else:
        raise FileNotFoundError(
            f"No se encontró directorio de imágenes para split={split}.\n"
            f"Probé:\n  {cand1}\n  {cand2}\n  {cand3}"
        )

    if not ann_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de anotaciones: {ann_file}")

    print(f"[INFO] Usando img_root = {img_root}")
    print(f"[INFO] Usando ann_file = {ann_file}")

    entries = []
    with ann_file.open("r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # Saltar líneas vacías
        if not line:
            i += 1
            continue

        # Si la línea es la ruta de una imagen (*.jpg)
        if ".jpg" in line:
            rel_path = line
            i += 1

            # Saltar líneas vacías entre ruta y num_boxes
            while i < n and not lines[i].strip():
                i += 1
            if i >= n:
                break

            # Línea con número de cajas
            num_line = lines[i].strip()
            try:
                num_boxes = int(num_line)
            except ValueError:
                # Si aquí no hay un entero, el archivo está raro → saltamos esta imagen
                i += 1
                continue
            i += 1

            bboxes = []
            boxes_read = 0

            # Leer hasta num_boxes líneas de cajas o hasta que aparezca otra imagen
            while i < n and boxes_read < num_boxes:
                box_line = lines[i].strip()
                i += 1

                if not box_line:
                    continue

                # Si aparece otra ruta de imagen antes de leer todas las cajas
                if ".jpg" in box_line:
                    # Retrocedemos un paso para que el while externo lo procese como nueva imagen
                    i -= 1
                    break

                parts = box_line.split()
                if len(parts) < 4:
                    continue

                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue

                x, y, w, h = vals[0], vals[1], vals[2], vals[3]
                invalid = int(vals[7]) if len(vals) > 7 else 0

                if invalid == 0 and w > 0 and h > 0:
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                    bboxes.append([x1, y1, x2, y2])

                boxes_read += 1

            img_path = img_root / rel_path
            entries.append({
                "img_path": str(img_path),
                "bboxes": np.array(bboxes, dtype=np.float32)
            })

        else:
            # Línea que no es .jpg ni vacía: la ignoramos
            i += 1

    print(f"[INFO] Entradas cargadas desde anotaciones: {len(entries)}")
    return entries



def compute_iou(box_a, box_b):
    """
    IoU entre dos cajas [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_detections_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Empareja cajas predichas con GT usando estrategia codiciosa.

    Retorna:
        tp, fp, fn
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        # No hay GT, todo lo predicho es FP
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        # No se predice nada, todo GT es FN
        return 0, 0, len(gt_boxes)

    gt_used = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for p in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, g in enumerate(gt_boxes):
            if gt_used[idx]:
                continue
            iou = compute_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            gt_used[best_idx] = True
        else:
            fp += 1

    fn = len(gt_boxes) - sum(gt_used)
    return tp, fp, fn




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
                "--format=csv,noheader,nounits"
            ],
            stderr=subprocess.DEVNULL
        )
        line = out.decode("utf-8").strip().splitlines()[0]
        mem_str, util_str = [x.strip() for x in line.split(",")]
        mem_mb = float(mem_str)
        util_pct = float(util_str)
        return mem_mb, util_pct
    except Exception:
        return None, None



def visualize_examples(examples):
    """
    Muestra ejemplos con cajas GT (verde) y predichas (rojo).
    Cierra con:
      - cualquier tecla -> pasa al siguiente ejemplo
      - q o ESC         -> sale del todo
      - cerrar con X    -> sale del todo
    """
    win_name = "YuNet vs WIDER"

    for ex in examples:
        img_path = ex["img_path"]
        gt_boxes = ex["gt_boxes"]
        pred_boxes = ex["pred_boxes"]

        img = cv2.imread(img_path)
        if img is None:
            continue

        # GT en verde
        for (x1, y1, x2, y2) in gt_boxes:
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

        # Predicciones en rojo
        for (x1, y1, x2, y2) in pred_boxes:
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, (0, 0, 255), 2)

        cv2.putText(img, "GT: verde | Pred: rojo",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow(win_name, img)
            key = cv2.waitKey(50) & 0xFF

            # Si se cerró la ventana con la X
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return

            if key != 255:  # se ha pulsado alguna tecla
                if key == 27 or key == ord('q'):  # ESC o q -> salir del todo
                    cv2.destroyAllWindows()
                    return
                else:
                    # cualquier otra tecla -> siguiente ejemplo
                    break

    cv2.destroyAllWindows()


def run_yunet_on_wider():
    root_dir = Path(ROOT_DIR)
    print(f"[INFO] Usando ROOT_DIR = {root_dir}")
    print(f"[INFO] Usando MODEL_PATH = {MODEL_PATH}")

    print(f"[INFO] Cargando anotaciones WIDER (split={SPLIT})...")
    entries = load_wider_annotations(root_dir, split=SPLIT)
    print(f"[INFO] Total de imágenes en anotaciones: {len(entries)}")

    if MAX_IMAGES is not None:
        entries = entries[:MAX_IMAGES]
        print(f"[INFO] Limitando a las primeras {len(entries)} imágenes para la prueba.")

    # Yunet
    print("[INFO] Inicializando YuNet (cv2.FaceDetectorYN_create)...")

    # Tamaño inicial dummy; luego se actualiza por imagen con setInputSize
    detector = cv2.FaceDetectorYN_create(
        model=MODEL_PATH,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    # Proceso y psutil
    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # priming

    # Métricas globales
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_faces = 0
    total_pred_faces = 0

    total_time = 0.0
    num_images_processed = 0

    cpu_samples = []
    ram_samples = []
    gpu_mem_samples = []
    gpu_util_samples = []

    # Ejemplos para visualización
    visual_examples = []

    # CSV por imagen
    csv_file = open(RESULTS_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "img_path",
        "num_gt_faces",
        "num_pred_faces",
        "tp",
        "fp",
        "fn",
        "proc_time_ms",
        "cpu_pct",
        "ram_mb",
        "gpu_mem_mb",
        "gpu_util_pct"
    ])

    t_global_start = time.time()

    for idx, entry in enumerate(entries):
        img_path = entry["img_path"]
        gt_boxes = entry["bboxes"]

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        # Muy importante: ajustar el input_size al tamaño de la imagen
        detector.setInputSize((w, h))

        t0 = time.time()
        retval, faces = detector.detect(img)
        t1 = time.time()

        proc_time = (t1 - t0)
        total_time += proc_time
        num_images_processed += 1

        # Construir cajas predichas a partir de YuNet
        pred_boxes = []
        if faces is not None and len(faces) > 0:
            # faces.shape = (N, 15): [x, y, w, h, score, l0x, l0y, ...]
            for f in faces:
                x, y, bw, bh = f[0:4]

                x1 = max(0.0, float(x))
                y1 = max(0.0, float(y))
                x2 = min(float(w - 1), x1 + max(0.0, float(bw)))
                y2 = min(float(h - 1), y1 + max(0.0, float(bh)))

                if x2 > x1 and y2 > y1:
                    pred_boxes.append([x1, y1, x2, y2])

        pred_boxes = np.array(pred_boxes, dtype=np.float32)

        tp, fp, fn = match_detections_to_gt(pred_boxes, gt_boxes, IOU_THRESHOLD)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt_faces += len(gt_boxes)
        total_pred_faces += len(pred_boxes)

        # Métricas de recursos
        cpu_pct = psutil.cpu_percent(interval=None)
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

        # Guardar por imagen en CSV
        csv_writer.writerow([
            img_path,
            len(gt_boxes),
            len(pred_boxes),
            tp,
            fp,
            fn,
            proc_time * 1000.0,
            cpu_pct,
            mem_mb,
            gpu_mem_mb if gpu_mem_mb is not None else "",
            gpu_util_pct if gpu_util_pct is not None else ""
        ])

        # Guardar algunos ejemplos para visualización
        if VISUAL_DEBUG and len(visual_examples) < MAX_VISUAL_EXAMPLES:
            if random.random() < VISUAL_SAMPLE_PROB:
                visual_examples.append({
                    "img_path": img_path,
                    "gt_boxes": gt_boxes.copy(),
                    "pred_boxes": pred_boxes.copy()
                })

        if num_images_processed % 100 == 0:
            print(f"[INFO] Procesadas {num_images_processed} imágenes...")

    t_global_end = time.time()
    csv_file.close()


    total_detections = total_tp + total_fp  # todas las cajas predichas

    precision = total_tp / total_detections if total_detections > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # "Accuracy" de detección: TP / (TP + FP + FN) (no hay TN bien definido)
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    elapsed = t_global_end - t_global_start
    avg_proc_time = total_time / num_images_processed if num_images_processed > 0 else 0.0
    fps = num_images_processed / total_time if total_time > 0 else 0.0

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0.0
    avg_gpu_mem = sum(gpu_mem_samples) / len(gpu_mem_samples) if gpu_mem_samples else None
    avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples) if gpu_util_samples else None

    print("\n========== RESULTADOS GLOBALES (YuNet + WIDER) ==========")
    print(f"ROOT_DIR:             {root_dir}")
    print(f"Split:                {SPLIT}")
    print(f"N imágenes procesadas:{num_images_processed}")
    print(f"Caras GT totales:     {total_gt_faces}")
    print(f"Caras predichas:      {total_pred_faces}")
    print(f"Caras correctamente detectadas (TP): {total_tp}")
    print(f"Falsos positivos (FP):               {total_fp}")
    print(f"Falsos negativos (FN):               {total_fn}")
    print("")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy*: {accuracy:.4f}  (*TP/(TP+FP+FN), sin TN)")
    print("")
    print(f"Tiempo total (wall clock): {elapsed:.2f} s")
    print(f"Tiempo total procesando imágenes (solo YuNet): {total_time:.2f} s")
    print(f"Tiempo medio por imagen (solo YuNet): {avg_proc_time*1000:.2f} ms")
    print(f"FPS aproximado (solo YuNet): {fps:.2f} imágenes/s")
    print("")
    print(f"CPU medio: {avg_cpu:.2f} %")
    print(f"RAM media: {avg_ram:.2f} MB")
    if avg_gpu_mem is not None:
        print(f"GPU memoria media: {avg_gpu_mem:.2f} MB")
        print(f"GPU uso medio:    {avg_gpu_util:.2f} %")
    else:
        print("GPU: no se pudieron obtener métricas (probablemente no hay nvidia-smi).")
    print("")
    print(f"Resultados por imagen guardados en: {RESULTS_CSV}")
    print("======================================================================")

    # Mostrar ejemplos visuales si se pidieron
    if VISUAL_DEBUG and visual_examples:
        print(f"[INFO] Mostrando {len(visual_examples)} ejemplos visuales...")
        visualize_examples(visual_examples)


if __name__ == "__main__":
    run_yunet_on_wider()
