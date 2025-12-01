# streaming/replay_mock_trials.py
import os
import sys

# Add project root to path so we can import backend code
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)



import time

import numpy as np

from config import FS, TRIAL_DURATION
from backend_client import send_trial_to_backend
from backend.mock_data import generate_mock_ssvep  # adjust import if needed


def main():
    """
    Simulate a live SSVEP stream by generating synthetic trials and sending
    them to the backend one by one, with timing that approximates real-time.
    """
    directions = {
        "up": 10.0,
        "down": 20.0,
        "left": 12.0,
        "right": 15.0,
    }

    n_trials_per_class = 3  # just for demo; you can increase this
    all_trials = []
    all_labels = []

    # 1. Generate trials for each direction (using the same generator as training)
    for direction, freq in directions.items():
        X_dir, y_dir = generate_mock_ssvep(
            direction=direction,
            freq=freq,
            n_trials=n_trials_per_class,
            fs=FS,
        )
        all_trials.append(X_dir)
        all_labels.append(y_dir)

    X_all = np.concatenate(all_trials, axis=0)  # shape: (N, channels, samples)
    y_all = np.concatenate(all_labels, axis=0)

    n_total_trials = X_all.shape[0]
    print("Prepared", n_total_trials, "synthetic trials")

    # 2. "Replay" each trial as if it just finished being collected
    trial_samples = int(TRIAL_DURATION * FS)
    print("Each trial:", trial_samples, "samples,", TRIAL_DURATION, "seconds")

    for i in range(n_total_trials):
        trial = X_all[i]      # shape: (channels, samples)
        true_label = y_all[i]

        print("\n--- Trial", i + 1, "of", n_total_trials, "---")
        print("True label:", true_label)

        # Simulate real-time trial collection: wait TRIAL_DURATION seconds
        print("Simulating data collection for", TRIAL_DURATION, "seconds...")
        time.sleep(TRIAL_DURATION)

        # 3. Send the full trial to backend /predict
        try:
            response = send_trial_to_backend(trial, model_name="lda")
        except Exception as e:
            print("Error sending trial to backend:", e)
            continue

        predicted = response.get("predicted_label", "UNKNOWN")
        probs = response.get("class_probabilities", {})

        print("Backend predicted:", predicted)
        print("Class probabilities:", probs)


if __name__ == "__main__":
    main()
