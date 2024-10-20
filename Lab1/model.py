import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def load_breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    return cancer

def preprocess_data(cancer):
    '''Нормализация (масштабирование) данных'''
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cancer_df)
    cancer_df_scaled = pd.DataFrame(scaled_data, columns=cancer.feature_names)
    cancer_df_scaled['target'] = cancer.target

    '''Исследовательский анализ данных'''
    data = cancer_df_scaled.groupby('target').mean().T
    data['diff'] = abs(data.iloc[:, 0] - data.iloc[:, 1])
    data = data.sort_values(by=['diff'], ascending=False)
    features = list(data.index[:10])
    X = cancer_df_scaled[features]
    y = cancer_df_scaled['target']
    return X, y

def split_data(X, y):
    '''Разделение модели на "обучающую" и "тестовую" части'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    '''Обучение и оценка качества модели'''
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    model_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, model_matrix, model_accuracy, report

def save_results(model_matrix, model_accuracy, report, file_path):
    '''Запись результатов в файл'''
    model_matrix_df = pd.DataFrame(model_matrix, columns=['Прогноз добр.', 'Прогноз злок.'], index=['Факт добр.', 'Факт злок.'])
    with open(file_path, "w") as file:
        file.write("Confusion Matrix:\n")
        file.write(model_matrix_df.to_string() + "\n\n")
        file.write("Accuracy: " + str(round(model_accuracy, 2)))
        file.write("\n\n" + "Report:\n")
        file.write(report)

if __name__ == "__main__":
    cancer = load_breast_cancer_data()
    X, y = preprocess_data(cancer)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model, model_matrix, model_accuracy, report = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    file_path = "predict.txt"
    save_results(model_matrix, model_accuracy, report, file_path)
    print("Classification results saved to", file_path)
