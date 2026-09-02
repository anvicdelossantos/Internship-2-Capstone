"""
train_model.py
----------------
DAY 3 DELIVERABLE: ML Model

This script does 3 things:
1. Generates a synthetic "driver behavior" dataset (stand-in for a real logistics
   dataset — swap this out with a real CSV later using pd.read_csv()).
2. Cleans/encodes the data and trains a RandomForestClassifier to predict a
   risk_level (low / medium / high) from driving telemetry.
3. Saves the trained model + the label encoder to model.pkl so the FastAPI
   service can load it later without retraining.

WHY RandomForest?
- Works well on small/medium tabular data without heavy tuning.
- Handles non-linear relationships (e.g. speeding matters more when combined
  with harsh braking, not just on its own).
- Gives us feature_importances_ for free, which is great for explaining
  *why* a driver was flagged as risky (useful for your capstone demo).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. GENERATE SYNTHETIC DATA
# ---------------------------------------------------------------------------
# Each row = one trip/segment by a driver. Features are things telematics
# hardware (GPS + accelerometer) or a fleet app can realistically capture.

N = 4000

def generate_dataset(n=N):
    avg_speed_kmh = np.random.normal(60, 20, n).clip(0, 140)
    speeding_events = np.random.poisson(1.5, n)
    harsh_braking_events = np.random.poisson(1.0, n)
    harsh_acceleration_events = np.random.poisson(0.8, n)
    sharp_turns = np.random.poisson(0.5, n)
    driving_hours = np.random.normal(6, 2.5, n).clip(0.5, 14)
    night_driving_pct = np.random.uniform(0, 1, n)
    phone_usage_events = np.random.poisson(0.3, n)
    fatigue_score = np.random.uniform(0, 1, n)  # 0=alert, 1=very fatigued

    # ---- risk score formula (ground truth logic used only to LABEL the
    # synthetic data — the model itself does NOT see this formula, it has to
    # learn the pattern from the features above) ----
    raw_risk = (
        0.015 * avg_speed_kmh
        + 0.9 * speeding_events
        + 1.1 * harsh_braking_events
        + 1.0 * harsh_acceleration_events
        + 0.7 * sharp_turns
        + 0.25 * driving_hours
        + 1.3 * night_driving_pct
        + 1.5 * phone_usage_events
        + 2.0 * fatigue_score
        + np.random.normal(0, 1.0, n)  # noise
    )

    # Convert continuous risk into 3 classes using percentile thresholds
    low_cut, high_cut = np.percentile(raw_risk, [40, 80])
    risk_level = np.where(raw_risk <= low_cut, "low",
                  np.where(raw_risk <= high_cut, "medium", "high"))

    df = pd.DataFrame({
        "avg_speed_kmh": avg_speed_kmh,
        "speeding_events": speeding_events,
        "harsh_braking_events": harsh_braking_events,
        "harsh_acceleration_events": harsh_acceleration_events,
        "sharp_turns": sharp_turns,
        "driving_hours": driving_hours,
        "night_driving_pct": night_driving_pct,
        "phone_usage_events": phone_usage_events,
        "fatigue_score": fatigue_score,
        "risk_level": risk_level,
    })
    return df


def main():
    df = generate_dataset()
    print("Dataset shape:", df.shape)
    print(df["risk_level"].value_counts())

    # -----------------------------------------------------------------
    # 2. PREPROCESSING
    # -----------------------------------------------------------------
    feature_cols = [
        "avg_speed_kmh", "speeding_events", "harsh_braking_events",
        "harsh_acceleration_events", "sharp_turns", "driving_hours",
        "night_driving_pct", "phone_usage_events", "fatigue_score",
    ]
    X = df[feature_cols]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["risk_level"])  # low=1, medium=2, high=0 (alphabetical)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # -----------------------------------------------------------------
    # 3. TRAIN MODEL
    # -----------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # 4. EVALUATE
    # -----------------------------------------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\nFeature importances:")
    for col, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {col:28s} {imp:.3f}")

    # -----------------------------------------------------------------
    # 5. SAVE MODEL + ENCODER + FEATURE ORDER
    # -----------------------------------------------------------------
    # We bundle everything the API needs into one dict so there's no
    # mismatch risk between training-time and inference-time feature order.
    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
    }
    joblib.dump(bundle, "model.pkl")
    print("\nSaved model.pkl")


if __name__ == "__main__":
    main()
