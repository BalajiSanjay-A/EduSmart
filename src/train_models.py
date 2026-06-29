import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score

PROCESSED_DATA_PATH = "data/processed_student_data.csv"
RAW_DATA_PATH = "data/student-mat.csv"
MODELS_DIR = "models"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

LINEAR_REG_PATH = "models/linear_reg.pkl"
LOGISTIC_REG_PATH = "models/logistic_reg.pkl"
KNN_PATH = "models/knn.pkl"
DECISION_TREE_PATH = "models/decision_tree.pkl"
RANDOM_FOREST_PATH = "models/random_forest.pkl"
KMEANS_PATH = "models/kmeans.pkl"
PCA_PATH = "models/pca.pkl"
RANDOM_STATE = 42
def load_processed_data():
    print(f"Loading processed dataset from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    print(f"Loaded {df.shape[0]} rows, {len(feature_columns)} features")
    return df, feature_columns

def load_raw_risk_inputs():
    raw_df = pd.read_csv(RAW_DATA_PATH, sep=None, engine='python')
    return raw_df[["failures", "absences"]]

def build_pass_fail_target(df):
    return (df["G3"] >= 10).astype(int)

def build_performance_tier_target(df):
    def tier(g3):
        if g3 < 10:
            return "Low"
        elif g3 < 14:
            return "Mid"
        else:
            return "High"
    return df["G3"].apply(tier)

def build_risk_level_target(raw_risk_inputs):
    def risk(row):
        failures, absences = row["failures"], row["absences"]
        if failures >= 2 or absences >= 20:
            return "High Risk"
        elif failures == 1 or absences >= 10:
            return "Medium Risk"
        else:
            return "Low Risk"
    return raw_risk_inputs.apply(risk, axis=1)

def train_linear_regression(X, y):
    print("\n--- [1/7] Linear Regression ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"R² score on test set: {r2:.3f}")

    joblib.dump(model, LINEAR_REG_PATH)
    print(f"Saved model to: {LINEAR_REG_PATH}")
    return model


def train_logistic_regression(X, y):
    print("\n--- [2/7] Logistic Regression ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy on test set: {acc:.3f}")
    print(f"Class balance in full dataset -> Pass: {(y == 1).sum()}, Fail: {(y == 0).sum()}")

    joblib.dump(model, LOGISTIC_REG_PATH)
    print(f"Saved model to: {LOGISTIC_REG_PATH}")
    return model


def train_knn(X):
    print("\n--- [3/7] KNN ---")
    model = NearestNeighbors(n_neighbors=5)
    model.fit(X)
    print(f"Fitted NearestNeighbors on {X.shape[0]} students using {X.shape[1]} features.")

    joblib.dump(model, KNN_PATH)
    print(f"Saved model to: {KNN_PATH}")
    return model


def train_decision_tree(X, y):
    print("\n--- [4/7] Decision Tree ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy on test set: {acc:.3f}")
    print(f"Class distribution in full dataset: {y.value_counts().to_dict()}")

    joblib.dump(model, DECISION_TREE_PATH)
    print(f"Saved model to: {DECISION_TREE_PATH}")
    return model


def train_random_forest(X, y):
    print("\n--- [5/7] Random Forest ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy on test set: {acc:.3f}")
    print(f"Class distribution in full dataset: {y.value_counts().to_dict()}")

    joblib.dump(model, RANDOM_FOREST_PATH)
    print(f"Saved model to: {RANDOM_FOREST_PATH}")
    return model


def train_kmeans(X):
    print("\n--- [6/7] K-Means ---")
    model = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = model.fit_predict(X)

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")

    joblib.dump(model, KMEANS_PATH)
    print(f"Saved model to: {KMEANS_PATH}")
    return model, cluster_labels


def train_pca(X):
    print("\n--- [7/7] PCA ---")
    model = PCA(n_components=2, random_state=RANDOM_STATE)
    components = model.fit_transform(X)

    explained = model.explained_variance_ratio_
    print(f"Explained variance ratio per component: {explained}")
    print(f"Total variance captured by 2 components: {explained.sum():.3f}")

    joblib.dump(model, PCA_PATH)
    print(f"Saved model to: {PCA_PATH}")
    return model, components


def run_training():
    df, feature_columns = load_processed_data()
    raw_risk_inputs = load_raw_risk_inputs()

    X = df[feature_columns]
    print(f"\nFeature matrix shape: {X.shape}")

    y_g3 = df["G3"]
    y_pass = build_pass_fail_target(df)
    y_tier = build_performance_tier_target(df)
    y_risk = build_risk_level_target(raw_risk_inputs)
    train_linear_regression(X, y_g3)
    train_logistic_regression(X, y_pass)
    train_knn(X)
    train_decision_tree(X, y_tier)
    train_random_forest(X, y_risk)
    train_kmeans(X)
    train_pca(X)

    print("\nAll 7 models trained and saved to the models/ folder!")


if __name__ == "__main__":
    run_training()
