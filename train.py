import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

original_data = pd.read_csv("Country_Dataset.csv")
#рид цсв
data = original_data.copy()

numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
X = data[numeric_columns].copy()
#входные данные

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#обучение и создание скалер

best_k = 2
best_score = -1
best_model = None
#переменные для поиска бест k

for k in range(2,3):
    model = KMeans(n_clusters=k, random_state=67, n_init=10)
    labels = model.fit_predict(X_scaled)
    #обуч модель и получ кластеры
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, silhouette_score={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k
        best_model = model
        #переменные для поиска бест k

data["Cluster"] = best_model.labels_
#адд кластер в табл
original_data["Cluster"] = best_model.labels_

print(f"\nBest number of clusters: {best_k}")
print(f"Best silhouette score: {best_score:.4f}")

print("\n10 samples with cluster labels:")
print(original_data.head(10))

print("\nAverage values by cluster:")
print(data.groupby("Cluster").mean())

if "country" in original_data.columns:
    print("\nReal countries in each cluster:")
    for i in range(best_k):
        print(f"\nCluster {i}:")
        countries = original_data[original_data["Cluster"] == i]["country"].tolist()
        for country in countries:
            print(country)

joblib.dump(best_model, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(numeric_columns, "feature_columns.pkl")
#сохранения

print("\nSaved:")
print("kmeans_model.pkl")
print("scaler.pkl")
print("feature_columns.pkl")
