# test_main.py
import pytest
from fastapi.testclient import TestClient
from api.main import app, HousingFeatures, PredictionResponse

# Inicjalizacja klienta testowego
client = TestClient(app)

# Przykładowe dane wejściowe
valid_input = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

partial_input = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": None,  # Brakująca wartość
    "Population": None,  # Brakująca wartość
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

invalid_input = {
    "MedInc": "invalid",  # Nieprawidłowy typ danych
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

# Testy

def test_predict_valid_input():
    """
    Test sprawdzający poprawność odpowiedzi dla prawidłowych danych wejściowych.
    """
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    assert "predicted_house_value" in response.json()
    assert isinstance(response.json()["predicted_house_value"], float)

def test_predict_partial_input():
    """
    Test sprawdzający poprawność odpowiedzi dla częściowo brakujących danych wejściowych.
    """
    response = client.post("/predict", json=partial_input)
    assert response.status_code == 200
    assert "predicted_house_value" in response.json()
    assert isinstance(response.json()["predicted_house_value"], float)

def test_predict_invalid_input():
    """
    Test sprawdzający obsługę nieprawidłowych danych wejściowych.
    """
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # Błąd walidacji Pydantic

def test_predict_missing_input():
    """
    Test sprawdzający obsługę całkowicie brakujących danych wejściowych.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 200

def test_predict_response_model():
    """
    Test sprawdzający, czy odpowiedź jest zgodna z modelem PredictionResponse.
    """
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    # Sprawdzenie, czy odpowiedź pasuje do modelu PredictionResponse
    try:
        PredictionResponse(**response.json())
    except Exception as e:
        pytest.fail(f"Odpowiedź nie pasuje do modelu PredictionResponse: {e}")

