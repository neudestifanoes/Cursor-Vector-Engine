# SSVEP Inference Backend

The backend is a machine-learning inference service built with **FastAPI**. It handles raw EEG data ingestion, feature extraction, and real-time classification using LDA and SVM models.

## Core Components

### 1. Synthetic EEG Generator (`mock_data.py`)
Used for development when a headset is unavailable. It mathematically recreates SSVEP signals:
* **Trial Structure:** 7 seconds total (2s baseline, 3s flicker stimulus, 2s rest).
* **Signal:** Adds sinusoidal waves at target frequencies (10, 12, 15, 20 Hz) plus harmonics (2x, 3x).
* **Format:** Generates 9 channels at 256 Hz (1792 samples per trial).

### 2. Feature Extraction (`features.py`)
Converts raw time-series data into ML-ready vectors.
* **Method:** Welch Power Spectral Density (PSD).
* **Features:** Extracts bandpower ±0.5 Hz around the target frequencies and harmonics.
* **Vector Size:** 108 features per trial (9 channels × 12 frequency components).

### 3. Model Training (`train_models.py`)
Trains and serializes the classifiers:
* **Models:** Linear Discriminant Analysis (LDA) and SVM (RBF kernel).
* **Output:** Saves models as `ssvep_lda_model.joblib` and `ssvep_svm_model.joblib`.

## API Endpoints (`app/main.py`)

The server exposes two primary endpoints:

### `GET /health`
Checks if the backend is running and models are loaded.
* **Response:** `{"status": "ok", "lda_loaded": true, "svm_loaded": true}`

### `POST /predict`
Accepts a raw EEG trial and returns the decoded direction.
* **Payload:**
    ```json
    {
      "data": [[channel1_samples], [channel2_samples], ...],
      "model_name": "lda"
    }
    ```
    *Data shape must be (9 channels × 1792 samples)*.

* **Response:**
    ```json
    {
      "model_name": "lda",
      "predicted_label": "up",
      "class_probabilities": {
        "up": 0.98,
        "down": 0.01,
        "left": 0.01,
        "right": 0.00
      }
    }
    ```
