import joblib
import pandas as pd

try:
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    print("Сначала запусти train.py")
    exit()

print("Введите значения признаков:")

input_values = []
#список для ввода

for col in feature_columns:
    value = float(input(f"{col}: ")) #ввод значения
    input_values.append(value) #адд в список

input_data = pd.DataFrame([input_values], columns=feature_columns) #ввод -> таблица
input_scaled = scaler.transform(input_data) #юс скалер
cluster = model.predict(input_scaled)[0] #получ кластер

print(f"\nPredicted cluster: {cluster}")
