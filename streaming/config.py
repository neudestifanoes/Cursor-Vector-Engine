# streaming/config.py

# URL of your backend FastAPI server
BACKEND_URL = "http://127.0.0.1:8000"

# Endpoint paths
PREDICT_ENDPOINT = "/predict"
HEALTH_ENDPOINT = "/health"

# SSVEP data properties (should match backend)
FS = 256          # sampling rate in Hz
N_CHANNELS = 9    # number of EEG channels
TRIAL_DURATION = 7.0  # seconds (2s baseline, 3s stim, 2s rest)
