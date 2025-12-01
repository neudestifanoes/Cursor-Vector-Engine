import numpy as np
from scipy.signal import welch


def extract_ssvep_features(
    X: np.ndarray,
    fs: int = 256,
    target_freqs: tuple[float, ...] = (10.0, 12.0, 15.0, 20.0),
    harmonic_multipliers: tuple[int, ...] = (1, 2, 3),
    band_width: float = 0.5,
) -> np.ndarray:
    """
    Extract SSVEP features from EEG data using band-power around tagged frequencies
    and their harmonics.

    Parameters
    ----------
    X : np.ndarray
        EEG data with shape (n_trials, n_channels, n_samples).
    fs : int
        Sampling rate in Hz. Default is 256.
    target_freqs : tuple of float
        Base stimulus frequencies in Hz (e.g., 10, 12, 15, 20).
    harmonic_multipliers : tuple of int
        Which harmonics to include (e.g., 1 = fundamental, 2 = 2f, 3 = 3f).
    band_width : float
        Half-width (in Hz) around each frequency for averaging PSD power.
        For example, band_width=0.5 → [f-0.5, f+0.5].

    Returns
    -------
    features : np.ndarray
        Feature matrix of shape (n_trials, n_features), where
        n_features = n_channels * (len(target_freqs) * len(harmonic_multipliers)).
    """
    n_trials, n_channels, n_samples = X.shape

    n_freq_components = len(target_freqs) * len(harmonic_multipliers)
    n_features = n_channels * n_freq_components

    features = np.zeros((n_trials, n_features), dtype=float)

    for trial_idx in range(n_trials):
        trial_feats = []

        for ch in range(n_channels):
            # 1D signal for this channel & trial
            sig = X[trial_idx, ch, :]

            # Welch PSD estimate: returns frequencies f and power spectral density psd
            f, psd = welch(sig, fs=fs, nperseg=min(512, n_samples))

            # For each base frequency and each harmonic, compute mean bandpower
            for base_f in target_freqs:
                for h in harmonic_multipliers:
                    freq = base_f * h
                    # Find indices around [freq - band_width, freq + band_width]
                    idx = np.where((f >= freq - band_width) & (f <= freq + band_width))[0]

                    if idx.size > 0:
                        power = psd[idx].mean()
                    else:
                        power = 0.0

                    trial_feats.append(power)

        features[trial_idx, :] = np.array(trial_feats)

    return features
