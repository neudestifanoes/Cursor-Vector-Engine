import numpy as np

def generate_mock_ssvep(
    direction: str,
    freq: float,
    n_trials: int = 20,
    n_channels: int = 8,
    duration: float = 7.0,
    fs: int = 256,
    noise_std: float = 0.3,
    random_state = None,
):
    """
    Generate synthetic EEG data for a Steady-State Visual Evoked Potential (SSVEP) experiment.

    This simulates realistic EEG recordings for a given visual stimulus direction
    and frequency. Each trial contains `n_channels` EEG channels with Gaussian noise
    and a sinusoidal SSVEP response added during the stimulus window (2–5 seconds).
    The output can be used for developing and testing BCI machine learning models.

    Parameters
    ----------
    direction : str
        Label for the visual stimulus direction (e.g., "up", "down", "left", "right").
    freq : float
        Frequency of the flickering visual stimulus in Hz (e.g., 10, 12, 15, 20).
    n_trials : int, optional
        Number of trials to generate for this direction. Default is 20.
    n_channels : int, optional
        Number of EEG channels (electrodes). Default is 8.
    duration : float, optional
        Total trial duration in seconds, including baseline (2s), stimulus (3s),
        and rest (2s). Default is 7 seconds.
    fs : int, optional
        Sampling rate in Hz. Default is 256 Hz.
    noise_std : float, optional
        Standard deviation of the Gaussian noise. Default is 0.3.
    random_state : int or None
        Optional seed for reproducibility.

    Returns
    -------
    X_data : np.ndarray
        Synthetic EEG dataset with shape (n_trials, n_channels, n_samples).
    y_labels : np.ndarray
        Array of string labels of shape (n_trials,), each equal to `direction`.
    """
    rng = np.random.default_rng(random_state)

    # Total samples in one trial: 7 s * 256 Hz = 1792 samples
    n_samples = int(duration * fs)

    # Time vector for the whole trial
    time = np.linspace(0, duration, n_samples, endpoint=False)

    # Indices for the stimulus window: 2s–5s (this matches your SSVEP protocol)
    stim_start = int(2 * fs)   # 2 * 256 = 512
    stim_end = int(5 * fs)     # 5 * 256 = 1280

    data = []
    for _ in range(n_trials):
        trial = []
        for _ in range(n_channels):
            # Background EEG-like noise for the full trial
            signal = rng.normal(0, noise_std, n_samples)

            # Base SSVEP sinusoid at the tagged frequency during stimulus window
            carrier = np.sin(2 * np.pi * freq * time[stim_start:stim_end])

            # Optional harmonics: 2f and 3f for more realistic SSVEP
            harmonic2 = 0.5 * np.sin(2 * np.pi * 2 * freq * time[stim_start:stim_end])
            harmonic3 = 0.25 * np.sin(2 * np.pi * 3 * freq * time[stim_start:stim_end])

            ssvep_segment = carrier + harmonic2 + harmonic3

            # Inject SSVEP only during the stimulus period
            signal[stim_start:stim_end] += ssvep_segment

            trial.append(signal)
        data.append(trial)

    # Final shapes: (trials, channels, samples)
    X_data = np.array(data)
    y_labels = np.array([direction] * n_trials)
    return X_data, y_labels


if __name__ == "__main__":
    # Frequencies for each direction (from your SSVEP protocol)
    directions = {
        "up": 10,
        "down": 20,
        "left": 12,
        "right": 15,
    }

    all_X = []
    all_y = []

    for direction, f in directions.items():
        X_dir, y_dir = generate_mock_ssvep(direction, f, n_trials=20, random_state=42)
        all_X.append(X_dir)
        all_y.append(y_dir)
        print(f"{direction}: X shape = {X_dir.shape}, y shape = {y_dir.shape}")

    # Stack all directions together: total trials = 4 * 20 = 80
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    print(f"\nAll data shape: {X_all.shape}, All labels shape: {y_all.shape}")

    # Optionally save for training scripts
    np.save("ssvep_mock_X.npy", X_all)
    np.save("ssvep_mock_y.npy", y_all)
    print("Saved ssvep_mock_X.npy and ssvep_mock_y.npy")
