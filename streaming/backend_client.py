# streaming/backend_client.py

import requests
import numpy as np

from config import BACKEND_URL, PREDICT_ENDPOINT


def send_trial_to_backend(trial_data, model_name="lda"):
    """
    Send a single EEG trial to the backend /predict endpoint.

    Parameters
    ----------
    trial_data : np.ndarray
        EEG array of shape (channels, samples)
    model_name : str
        "lda" or "svm"

    Returns
    -------
    dict
        JSON response from the backend (parsed as a Python dict).
    """
    if isinstance(trial_data, np.ndarray):
        if trial_data.ndim != 2:
            raise ValueError("trial_data must be 2D: channels x samples")
        data_list = trial_data.tolist()
    else:
        # assume already a nested list
        data_list = trial_data

    payload = {
        "data": data_list,
        "model_name": model_name,
    }

    url = BACKEND_URL + PREDICT_ENDPOINT
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()
