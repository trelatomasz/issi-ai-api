# %%
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

data_set = "california_housing"
estimator="random_forest"

model_file_name=f"trained_models/{data_set}_{estimator}.pkl"
model_mean_features_name= f"trained_models/{data_set}_{estimator}_mean_values.pkl"
def load_model():
    return joblib.load(f'{model_file_name}')

def train():
    import time
    from datetime import datetime
    from timeit import timeit
    import joblib

    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    # Wczytanie danych California Housing
    time.time()
    california = fetch_california_housing()

    fetch_start = time.time()
    X, y = california.data, california.target
    fetch_end = time.time()
    print(f"Pobieranie datasetu {fetch_end - fetch_start:.4f} sekund.")

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Stworzenie regresora: Random Forest Regressor
    #
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Uzupełnianie brakujących wartości średnią
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])

    def train_model():
        model.fit(X_train, y_train)

    training_time = timeit(train_model, number=1)

    print(f"Trenowanie modelu zajeło {training_time:.4f} sekund.")
    print(f"Zapisuje model do pliku: {model_file_name}")
    joblib.dump(model, model_file_name )

    # Obliczenie średnich wartości dla każdej cechy
    # feature_means = np.mean(X_train, axis=0)
    # joblib.dump(feature_means, model_mean_features_name)

    # Przewidywanie na danych testowych
    y_pred = model.predict(X_test)

    # Ocena modelu
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Rzeczywista wartość')
    plt.ylabel('Przewidywana wartość')
    plt.title('California Housing: Rzeczywiste vs Przewidywane wartości')
    plt.grid(True)

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"results/{data_set}_{estimator}_{date}.png")



if __name__ == "__main__":
    train()