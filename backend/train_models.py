import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

from mock_data import generate_mock_ssvep
from features import extract_ssvep_features


def build_dataset_from_mock(
    n_trials_per_class: int = 50,
    fs: int = 256,
):
    """
    Generate a synthetic SSVEP dataset for all four directions using the mock generator.
    """
    directions = {
        "up": 10.0,
        "down": 20.0,
        "left": 12.0,
        "right": 15.0,
    }

    X_list = []
    y_list = []

    for direction, f in directions.items():
        X_dir, y_dir = generate_mock_ssvep(
            direction=direction,
            freq=f,
            n_trials=n_trials_per_class,
            fs=fs,
            random_state=42,
        )
        X_list.append(X_dir)
        y_list.append(y_dir)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


def train_and_save_models():
    # 1. Build dataset (you can also load from saved .npy if you want)
    X_raw, y = build_dataset_from_mock(n_trials_per_class=50)
    print(f"Raw data shape: {X_raw.shape}, labels shape: {y.shape}")

    # 2. Extract bandpower features
    X = extract_ssvep_features(X_raw, fs=256)
    print(f"Feature matrix shape: {X.shape}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # 4. Pipelines: StandardScaler + classifier
    lda_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, gamma="scale")),
    ])

    # 5. Train
    lda_pipeline.fit(X_train, y_train)
    svm_pipeline.fit(X_train, y_train)

    # 6. Evaluate
    for name, model in [("LDA", lda_pipeline), ("SVM", svm_pipeline)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=["up", "down", "left", "right"])
        print(f"\n{name} accuracy: {acc:.3f}")
        print(f"{name} confusion matrix (rows=true, cols=pred):\n{cm}")

    # 7. Save models
    joblib.dump(lda_pipeline, "ssvep_lda_model.joblib")
    joblib.dump(svm_pipeline, "ssvep_svm_model.joblib")
    print("\nSaved ssvep_lda_model.joblib and ssvep_svm_model.joblib")


if __name__ == "__main__":
    train_and_save_models()
