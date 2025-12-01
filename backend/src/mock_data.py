import numpy as np

def generate_mock_ssvep(direction, freq, n_trials=20, n_channels=9, duration=7, fs=256):
    """
    Generate synthetic EEG data for a Steady-State Visual Evoked Potential (SSVEP) experiment.

    This function simulates realistic EEG recordings for a given visual stimulus direction
    and frequency. Each trial contains 9 EEG channels with Gaussian noise and a sinusoidal
    SSVEP response added during the stimulus window (2–5 seconds). The output can be used 
    for developing and testing BCI machine learning models.

    Parameters
    ----------
    direction : str
        Label for the visual stimulus direction (e.g., "up", "down", "left", "right").
    freq : float
        Frequency of the flickering visual stimulus in Hz (e.g., 10, 12, 15, 20).
    n_trials : int, optional
        Number of trials to generate for this direction. Default is 20.
    n_channels : int, optional
        Number of EEG channels (electrodes). Default is 9.
    duration : float, optional
        Total trial duration in seconds, including baseline (2s), stimulus (3s),
        and rest (2s). Default is 7 seconds.
    fs : int, optional
        Sampling rate in Hz. Default is 256 Hz.

    Returns
    -------
    X_data : np.ndarray
        Synthetic EEG dataset with shape `(n_trials, n_channels, n_samples)`.
        Each trial contains 9 channels of continuous EEG data.
    y_labels : np.ndarray
        Array of string labels of shape `(n_trials,)`, each equal to the given direction.

    Notes
    -----
    - The simulated signal adds a sinusoidal component between 2s and 5s 
      (corresponding to indices 512:1280 for fs=256).
    - Noise is drawn from a Gaussian distribution (mean 0, std 0.3) to resemble
      real EEG background activity.
    - Can be extended with phase shifts, inter-trial variability, or realistic 
      noise patterns for advanced simulation.

    Examples
    --------
    >>> X_up, y_up = generate_mock_ssvep("up", 10)
    >>> X_up.shape (20, 9, 1792)
    >>> y_up[:5]
    array(['up', 'up', 'up', 'up', 'up'], dtype='<U2')
    """

    n_samples = int(duration * fs)
    time = np.linspace(0, duration, n_samples)
    
    # Simulate EEG-like sinusoidal signals + noise
    data = []
    for _ in range(n_trials):
        trial = []
        for _ in range(n_channels):
            # SSVEP signal during stimulus window (2s–5s)
            signal = np.random.normal(0, 0.3, n_samples)
            signal[512:1280] += np.sin(2 * np.pi * freq * time[512:1280])  # 3s stimulus
            trial.append(signal)
        data.append(trial)
    
    X_data = np.array(data)  # (n_trials, n_channels, n_samples)
    y_labels = np.array([direction] * n_trials)
    return X_data, y_labels

if __name__ == "__main__":
        
    # Parameters from the Synapse X study
    directions = {
        "up": 10,
        "down": 20,
        "left": 12,
        "right": 15
    }

    # Generate 4 separate datasets, one for each direction
    # Each X_data shape: (n_trials, n_channels, n_samples) (20, 9, 1792)
    # Each y_labels shape: (n_trials,) (20,)
    X_up_data, y_up_labels = generate_mock_ssvep("up", 10)
    X_down_data, y_down_labels = generate_mock_ssvep("down", 20)
    X_left_data, y_left_labels = generate_mock_ssvep("left", 12)
    X_right_data, y_right_labels = generate_mock_ssvep("right", 15)

    print("Mock SSVEP data generated:")
    print(f"Up data shape: {X_up_data.shape}, Labels shape: {y_up_labels.shape}")
    print(f"Up Data: {X_up_data}")
    
    # print(f"Down data shape: {X_down_data.shape}, Labels shape: {y_down_labels.shape}")
    # print(f"Left data shape: {X_left_data.shape}, Labels shape: {y_left_labels.shape}")
    # print(f"Right data shape: {X_right_data.shape}, Labels shape: {y_right_labels.shape}")
