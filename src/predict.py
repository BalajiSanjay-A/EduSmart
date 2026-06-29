import numpy as np
import pandas as pd
import joblib

PROCESSED_DATA_PATH = "data/processed_student_data.csv"
RAW_DATA_PATH = "data/student-mat.csv"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/label_encoders.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
LINEAR_REG_PATH = "models/linear_reg.pkl"
LOGISTIC_REG_PATH = "models/logistic_reg.pkl"
KNN_PATH = "models/knn.pkl"
DECISION_TREE_PATH = "models/decision_tree.pkl"
RANDOM_FOREST_PATH = "models/random_forest.pkl"
KMEANS_PATH = "models/kmeans.pkl"
PCA_PATH = "models/pca.pkl"
def build_cluster_label_map(artifacts):
    kmeans_model = artifacts["kmeans"]
    processed_df = artifacts["processed_df"]
    feature_columns = artifacts["feature_columns"]
    X = processed_df[feature_columns]
    cluster_assignments = kmeans_model.predict(X)
    temp_df = processed_df.copy()
    temp_df["cluster"] = cluster_assignments
    avg_g3_per_cluster = temp_df.groupby("cluster")["G3"].mean()
    sorted_clusters = avg_g3_per_cluster.sort_values().index.tolist()
    cluster_label_map = {
        sorted_clusters[0]: "Needs Improvement",
        sorted_clusters[1]: "Average",
        sorted_clusters[2]: "High Performer",
    }
    return cluster_label_map

def load_all_artifacts():
    artifacts = {}
    artifacts["scaler"] = joblib.load(SCALER_PATH)
    artifacts["encoders"] = joblib.load(ENCODERS_PATH)
    artifacts["feature_columns"] = joblib.load(FEATURE_COLUMNS_PATH)
    artifacts["linear_reg"] = joblib.load(LINEAR_REG_PATH)
    artifacts["logistic_reg"] = joblib.load(LOGISTIC_REG_PATH)
    artifacts["knn"] = joblib.load(KNN_PATH)
    artifacts["decision_tree"] = joblib.load(DECISION_TREE_PATH)
    artifacts["random_forest"] = joblib.load(RANDOM_FOREST_PATH)
    artifacts["kmeans"] = joblib.load(KMEANS_PATH)
    artifacts["pca"] = joblib.load(PCA_PATH)
    artifacts["processed_df"] = pd.read_csv(PROCESSED_DATA_PATH)
    artifacts["raw_df"] = pd.read_csv(RAW_DATA_PATH, sep=None, engine='python')
    artifacts["cluster_label_map"] = build_cluster_label_map(artifacts)
    return artifacts

ARTIFACTS = load_all_artifacts()

def encode_raw_input(raw_input, encoders):
    encoded = dict(raw_input)
    for col, encoder in encoders.items():
        if col not in encoded:
            continue
        raw_value = encoded[col]
        if isinstance(encoder, dict):
            encoded[col] = encoder[raw_value]
        else:
            encoded[col] = int(encoder.transform([raw_value])[0])
    return encoded

def build_scaled_feature_row(raw_input, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    encoders = artifacts["encoders"]
    feature_columns = artifacts["feature_columns"]
    scaler = artifacts["scaler"]
    encoded_input = encode_raw_input(raw_input, encoders)
    input_df = pd.DataFrame([encoded_input])[feature_columns]
    scaled_values = scaler.transform(input_df)
    scaled_row = pd.DataFrame(scaled_values, columns=feature_columns)
    return scaled_row

def predict_final_grade(scaled_row, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    model = artifacts["linear_reg"]
    raw_prediction = model.predict(scaled_row)[0]
    clipped = float(np.clip(raw_prediction, 0, 20))
    return round(clipped, 2)

def predict_pass_fail(scaled_row, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    model = artifacts["logistic_reg"]
    prediction = model.predict(scaled_row)[0]
    return "Pass" if prediction == 1 else "Fail"

def predict_performance_tier(scaled_row, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    model = artifacts["decision_tree"]
    return model.predict(scaled_row)[0]

def predict_risk_level(scaled_row, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    model = artifacts["random_forest"]
    return model.predict(scaled_row)[0]

def get_all_predictions(raw_input, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    scaled_row = build_scaled_feature_row(raw_input, artifacts)
    return {
        "predicted_grade": predict_final_grade(scaled_row, artifacts),
        "pass_fail": predict_pass_fail(scaled_row, artifacts),
        "performance_tier": predict_performance_tier(scaled_row, artifacts),
        "risk_level": predict_risk_level(scaled_row, artifacts),
    }
def find_similar_students(raw_input, artifacts=None, n_neighbors=5):
    if artifacts is None:
        artifacts = ARTIFACTS
    knn_model = artifacts["knn"]
    raw_df = artifacts["raw_df"]
    scaled_row = build_scaled_feature_row(raw_input, artifacts)
    distances, indices = knn_model.kneighbors(scaled_row, n_neighbors=n_neighbors)
    similar_students = []
    for distance, idx in zip(distances[0], indices[0]):
        student_record = raw_df.iloc[idx].to_dict()
        student_record["distance"] = round(float(distance), 3)
        student_record["row_index"] = int(idx)
        similar_students.append(student_record)
    return similar_students
def predict_cluster(raw_input, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    kmeans_model = artifacts["kmeans"]
    cluster_label_map = artifacts["cluster_label_map"]
    scaled_row = build_scaled_feature_row(raw_input, artifacts)
    cluster_number = int(kmeans_model.predict(scaled_row)[0])
    return {
        "cluster_number": cluster_number,
        "cluster_label": cluster_label_map[cluster_number],
    }
def project_to_pca(raw_input, artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    pca_model = artifacts["pca"]
    scaled_row = build_scaled_feature_row(raw_input, artifacts)
    coordinates = pca_model.transform(scaled_row)[0]
    return {"pc1": float(coordinates[0]), "pc2": float(coordinates[1])}

def get_dataset_pca_projection(artifacts=None):
    if artifacts is None:
        artifacts = ARTIFACTS
    pca_model = artifacts["pca"]
    processed_df = artifacts["processed_df"]
    feature_columns = artifacts["feature_columns"]
    X = processed_df[feature_columns]
    coordinates = pca_model.transform(X)
    result_df = pd.DataFrame(coordinates, columns=["pc1", "pc2"])
    result_df["G3"] = processed_df["G3"].values
    return result_df
if __name__ == "__main__":
    print("Testing predict.py...\n")
    sample_raw_row = ARTIFACTS["raw_df"].iloc[0]
    sample_input = sample_raw_row.drop(labels=["G1", "G2", "G3"]).to_dict()
    print("Predictions:")
    predictions = get_all_predictions(sample_input)
    for key, value in predictions.items():
        print(f"  {key}: {value}")
    print(f"\nActual G3: {sample_raw_row['G3']}")
    print("\nSimilar students:")
    similar = find_similar_students(sample_input)
    for s in similar:
        print(f"  row {s['row_index']} | dist={s['distance']} | G3={s['G3']}")
    print(f"\nCluster: {predict_cluster(sample_input)}")
    print(f"\nPCA: {project_to_pca(sample_input)}")
