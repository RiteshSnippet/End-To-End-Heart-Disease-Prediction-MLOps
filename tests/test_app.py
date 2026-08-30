import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")

    assert response.status_code == 200


def test_404_page(client):
    response = client.get("/wrong-page")

    assert response.status_code == 404


def test_prediction_page(client):
    response = client.post(
        "/",
        data={
            "age": "43",
            "sex": "1",
            "cp": "0",
            "trestbps": "150",
            "chol": "247",
            "fbs": "0",
            "restecg": "1",
            "thalach": "171",
            "exang": "0",
            "oldpeak": "1.5",
            "slope": "2",
            "ca": "0",
            "thal": "2"
        }
    )

    assert response.status_code == 200
