import requests

def test_health_check():
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction():
    data = {"feature1": 0.5}  # Example input, adapt as necessary
    response = requests.post("http://localhost:8000/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
