import numpy as np
import pandas as pd
from model import load_breast_cancer_data, preprocess_data, split_data, train_and_evaluate_model, save_results
from sklearn.preprocessing import StandardScaler

feature_data = [
    {"name": "mean radius", "interval": (6.98, 28.11)},
    {"name": "mean texture", "interval": (9.71, 39.28)},
    {"name": "mean perimeter", "interval": (43.79, 188.5)},
    {"name": "mean area", "interval": (143.5, 2501)},
    {"name": "mean smoothness", "interval": (0.05, 0.16)},
    {"name": "mean compactness", "interval": (0.02, 0.35)},
    {"name": "mean concavity", "interval": (0, 0.43)},
    {"name": "mean concave points", "interval": (0, 0.2)},
    {"name": "mean symmetry", "interval": (0.11, 0.3)},
    {"name": "mean fractal dimension", "interval": (0.05, 0.1)},
    {"name": "radius error", "interval": (0.11, 2.87)},
    {"name": "texture error", "interval": (0.36, 4.88)},
    {"name": "perimeter error", "interval": (0.76, 21.98)},
    {"name": "area error", "interval": (6.8, 542.2)},
    {"name": "smoothness error", "interval": (0, 0.03)},
    {"name": "compactness error", "interval": (0, 0.14)},
    {"name": "concavity error", "interval": (0, 0.4)},
    {"name": "concave points error", "interval": (0, 0.05)},
    {"name": "symmetry error", "interval": (0.01, 0.08)},
    {"name": "fractal dimension error", "interval": (0, 0.03)},
    {"name": "worst radius", "interval": (7.93, 36.04)},
    {"name": "worst texture", "interval": (12.02, 49.54)},
    {"name": "worst perimeter", "interval": (50.41, 251.2)},
    {"name": "worst area", "interval": (185.2, 4254)},
    {"name": "worst smoothness", "interval": (0.07, 0.22)},
    {"name": "worst compactness", "interval": (0.03, 1.06)},
    {"name": "worst concavity", "interval": (0, 1.25)},
    {"name": "worst concave points", "interval": (0, 0.29)},
    {"name": "worst symmetry", "interval": (0.16, 0.66)},
    {"name": "worst fractal dimension", "interval": (0.06, 0.21)}
]

def generate_synthetic_data(num_samples=569, num_features=30):
    np.random.seed(42)
    synthetic_cancer_data = {}

    for i in range(num_features):
        feature_info = feature_data[i % len(feature_data)]
        feature_name = feature_info["name"]
        low, high = feature_info["interval"]
        random_values = np.random.uniform(low, high, num_samples)
        synthetic_cancer_data[feature_name] = random_values

    synthetic_cancer_data['target'] = np.random.choice([0, 1], size=num_samples)
    synthetic_cancer_df = pd.DataFrame(synthetic_cancer_data)

    return synthetic_cancer_df

def preprocess_synthetic_data(data):
    X = data[[feature_info["name"] for feature_info in feature_data]]
    y = data['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

cancer = load_breast_cancer_data()
X, y = preprocess_data(cancer)
X_train, X_test, y_train, y_test = split_data(X, y)
model, model_matrix, model_accuracy, report = train_and_evaluate_model(X_train, y_train, X_test, y_test)
file_path = "predict.txt"
save_results(model_matrix, model_accuracy, report, file_path)
print("Classification results saved to", file_path)

synthetic_cancer_df = generate_synthetic_data()
X_synthetic, y_synthetic = preprocess_synthetic_data(synthetic_cancer_df)
X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = split_data(X_synthetic, y_synthetic)
model_synthetic, model_matrix_synthetic, model_accuracy_synthetic, report_synthetic = train_and_evaluate_model(X_train_synthetic, y_train_synthetic, X_test_synthetic, y_test_synthetic)
file_path_synthetic = "synthetic_results.txt"
save_results(model_matrix_synthetic, model_accuracy_synthetic, report_synthetic, file_path_synthetic)
print("Synthetic classification results saved to", file_path_synthetic)
