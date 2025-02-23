# issi-ai-api

Prosty serwis API z wykorzystaniem biblioteki FastAPI.

# Zasady użycia

## Trenowanie modelu i zapis do pliku
`python model/california_housing.py`

## Uruchom serwer: 
`fastapi dev api/main.py`
lub
`uvicorn api.main:app --reload`

## Testownie modelu

http://localhost:8000/docs#/default/predict_predict_post