import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

RAW_DATA_PATH = "data/student-mat.csv"
PROCESSED_DATA_PATH = "data/processed_student_data.csv"
MODELS_DIR = "models"
SCALER_SAVE_PATH = "models/scaler.pkl"
ENCODERS_SAVE_PATH = "models/label_encoders.pkl"
FEATURE_COLUMNS_SAVE_PATH = "models/feature_columns.pkl"

def load_data(filepath=RAW_DATA_PATH):
    print(f"[1/5] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath, sep=None, engine='python')
    print(f"      Loaded dataset with shape: {df.shape}")
    return df

def encode_features(df):
    df = df.copy()
    encoders = {}
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"[2/5] Found text columns to process: {text_columns}")
    
    for col in text_columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals == {"yes", "no"} or unique_vals == {"yes"} or unique_vals == {"no"}:
            yes_no_map = {"yes": 1, "no": 0}
            df[col] = df[col].map(yes_no_map)
            encoders[col] = yes_no_map
        elif len(unique_vals) == 2:
            sorted_vals = sorted(list(unique_vals))
            mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
            df[col] = df[col].map(mapping)
            encoders[col] = mapping
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders

def scale_features(df):
    target_cols = ["G1", "G2", "G3"]
    existing_targets = [col for col in target_cols if col in df.columns]
    feature_columns = [col for col in df.columns if col not in existing_targets]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_columns])
    scaled_df = pd.DataFrame(scaled_values, columns=feature_columns, index=df.index)
    for col in existing_targets:
        scaled_df[col] = df[col].values
    print(f"[3/5] Scaled {len(feature_columns)} feature columns")
    return scaled_df, scaler, feature_columns

def save_artifacts(processed_df, scaler, encoders, feature_columns):
    os.makedirs(MODELS_DIR, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[4/5] Saved processed dataset to: {PROCESSED_DATA_PATH}")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(encoders, ENCODERS_SAVE_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_SAVE_PATH)
    print(f"[5/5] All preprocessing artifacts saved!")

def run_preprocessing():
    df = load_data()
    encoded_df, encoders = encode_features(df)
    scaled_df, scaler, feature_columns = scale_features(encoded_df)
    save_artifacts(scaled_df, scaler, encoders, feature_columns)
    print("\nPreprocessing complete without column hardcoding issues!")
    return scaled_df, scaler, encoders, feature_columns

if __name__ == "__main__":
    run_preprocessing()