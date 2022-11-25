import json
from fastapi.testclient import TestClient
from typing import List
from .scr.utils import load_model

from online_ml_project.app import app
from .app import make_predict

client = TestClient(app)
data_example = [
    [69, 1, 0, 160, 234, 1, 2, 0, 1, 1, 1, 0],
    [35, 1, 3, 126, 282, 0, 2, 1, 0, 0, 0, 2]
]
model_example = load_model('data/model.joblib')


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == 'Welcome to the model hub!'


def test_make_prediction():
    preds = make_predict(data_example,
                         model_=model_example)
    assert isinstance(preds, List)


def test_correct_predict():
    response = client.post(
        "/predict",)
    # request = data_example),
    # responce_content = response.json()
    # assert expected_status == response.status_code
    # assert len(responce_content) == len(get_test_request_data)
