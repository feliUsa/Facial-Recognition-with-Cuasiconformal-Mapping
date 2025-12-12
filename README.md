# Facial Recognition with Cuasiconformal Mapping

Repositorio de experimentación para **detección/identificación/reconocimiento facial** incorporando una etapa de **normalización geométrica mediante mapeo cuasiconforme (QC)** y comparación con distintos enfoques/librerías de reconocimiento.

> Estructura principal del repo:
> - `cuasiconforme/`: implementación/experimentos del mapeo cuasiconforme y normalización geométrica.
> - `identificacion/`: experimentos de identificación/verificación con librerías y pipelines base.
> - `reconocimiento/`: experimentos de reconocimiento (embeddings + matching), medición de métricas y pruebas en tiempo real.

---

## Objetivo

1. Normalizar la geometría facial (QC) para reducir variabilidad por pose/expresión.
2. Evaluar librerías/modelos de reconocimiento facial con métricas cuantitativas (accuracy, tiempos, robustez temporal, etc.).
3. Proveer scripts reproducibles para ejecutar pruebas en **tiempo real** con webcam y/o datasets.

---

## Requisitos

- Ubuntu (recomendado) / Linux
- Python 3.10+ (ideal 3.10/3.11 por compatibilidad de paquetes)
- Webcam (para modo real-time)
- Dependencias típicas según el módulo:
  - `opencv-python` / `opencv-contrib-python`
  - `mediapipe`
  - `numpy`
  - `psutil`
  - Modelos de reconocimiento según experimento:
    - `insightface` (Buffalo_L)
    - `deepface`
    - `facenet-pytorch`
    - `face_recognition` (dlib)

> Recomendación fuerte: **usar entornos virtuales separados** por familia de dependencias (especialmente si mezclas TensorFlow/DeepFace, MediaPipe, InsightFace, PyTorch).

---

## Instalación rápida (base)

```bash
git clone https://github.com/feliUsa/Facial-Recognition-with-Cuasiconformal-Mapping.git
cd Facial-Recognition-with-Cuasiconformal-Mapping

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools


## Datasets utilizados (entrenamiento y pruebas)

Para los experimentos se usaron datasets públicos de Kaggle y se organizaron resultados en **dataframes** (CSV/estructuras tabulares) para entrenamientos, pruebas y cálculo de métricas.

### Asignación por módulo

- **`cuasiconforme/`**
  - **Celebrity Faces Dataset (Kaggle)**: usado para pruebas relacionadas con el flujo de normalización geométrica y evaluación del mapeo cuasiconforme sobre rostros.

- **`identificacion/`**
  - **WIDER FACE (wider-face-detection, Kaggle)**: usado principalmente para escenarios de **detección/identificación** con alta variabilidad (escala, oclusión, iluminación) y para evaluar robustez.

- **`reconocimiento/`**
  - **LFW (lfw-dataset, Kaggle)**: usado para pruebas de **verificación/reconocimiento** (comparación de embeddings, umbrales de similitud y métricas agregadas).

> Nota: Los dataframes se emplean para estructurar etiquetas/predicciones, scores (similitud/distancia), tiempos por etapa (detección/embedding/total), FPS y métricas globales (accuracy, precision/recall, etc.).
