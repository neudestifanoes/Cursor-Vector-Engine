# Neural Cursor Control (SSVEP BCI)

This repository contains a **brain-computer interface (BCI)** designed to decode **steady-state visually evoked potentials (SSVEP)** from EEG signals to control a directional cursor in real-time.

The system is designed around an **OpenBCI Cyton + Ultracortex Mark IV** setup and currently operates using a high-fidelity synthetic data generator for development and testing.

## 🏗 System Architecture

The project is divided into three distinct layers functioning as an end-to-end pipeline:

1.  **Visual Stimulus (Frontend):** The user focuses on flickering targets (Up/Down/Left/Right) blinking at specific frequencies (10, 12, 15, 20 Hz).
2.  **Streaming Layer:** Handles data acquisition. Currently, it replays synthetic trials to simulate a live EEG headset. Future integration will connect directly to the OpenBCI Cyton board.
3.  **Backend (Analysis):** Receives raw EEG, extracts Welch PSD features, and uses trained LDA/SVM models to predict the user's intent.

## 📂 Repository Structure

* **`backend/`**: FastAPI server, feature extraction logic, and Machine Learning models (LDA/SVM).
* **`frontend/`**: React-based dashboard for system health, visual stimuli, and cursor control.
* **`streaming/`**: Scripts for mocking "live" EEG data streams and replaying trials.

## 🚀 Quick Start

To run the full stack, you will need to start the backend server and the frontend client simultaneously.

**1. Start the Backend**
Navigate to `backend/` and run the FastAPI server (see `backend/README.md` for details).

**2. Start the Frontend**
Navigate to `frontend/` and start the React application (see `frontend/README.md` for details).

**3. Run a Simulation**
Navigate to `streaming/` to replay mock trials against the running backend (see `streaming/README.md` for details).