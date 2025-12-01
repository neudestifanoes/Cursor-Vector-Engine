# Streaming Layer (Mock & Live EEG)

The streaming layer handles the data acquisition pipeline. Currently, it simulates an online experiment by replaying synthetic trials, acting as a stand-in for the OpenBCI Cyton board.

## Components

### `config.py`
Centralizes shared settings to ensure consistency with the backend.
* **Sample Rate (FS):** 256 Hz.
* **Channels:** 9.
* **Trial Duration:** 7 seconds.
* **Backend URL:** Defaults to `http://127.0.0.1:8000`.

### `backend_client.py`
A wrapper for the backend API. It handles data formatting, serialization, and error handling for `POST /predict` requests. It allows the streaming logic to simply "send a trial" without manually constructing HTTP requests.

### `replay_mock_trials.py`
Simulates a live session.
1.  **Generates Data:** Creates synthetic SSVEP trials for all 4 classes (10, 12, 15, 20 Hz).
2.  **Simulates Time:** Waits `TRIAL_DURATION` (7 seconds) between requests to mimic real-time data collection.
3.  **Predicts:** Sends the data to the backend and prints the returned prediction and confidence scores.

## Usage

To test the system without a headset:

1.  Ensure the **Backend** is running.
2.  Run the replay script:
    ```bash
    python replay_mock_trials.py
    ```
3.  Observe the console output for "Backend predicted: <label>" validation.

## 🔜 Next Steps
The architecture is designed so that `replay_mock_trials.py` can be replaced with an LSL or serial stream reader for the OpenBCI Cyton without changing the rest of the pipeline.
