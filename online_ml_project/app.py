import uvicorn
from fastapi import FastAPI
from typing import Optional

from app_params import XInput
from model_utils.utils import load_model, SklearnClassifierModel

app = FastAPI()
model: Optional[SklearnClassifierModel] = None


def predict(data: list, model_: SklearnClassifierModel):
    return model_.predict(data)


@app.get('/')
async def main():
    return 'Welcome to the model hub!'


@app.get('/predict')
async def predict(request: XInput):
    global pretrained_model
    pretrained_model = load_model('/data/model.joblib')
    return pretrained_model.predict(request)


# @app.on_event('startup')
# def get_model():
#     global pretrained_model
#     pretrained_model = load_model('data/model.joblib')


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000)
