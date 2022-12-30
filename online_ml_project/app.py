import uvicorn
import logging
from fastapi import FastAPI
from typing import Optional, List

from app_params import XInput, ModelResponse
from scr.utils import load_model, SklearnClassifierModel
from scr.logger import initialize_logger

app_ = FastAPI()
logger = logging.getLogger(__name__)
logger = initialize_logger(logger, level=2)
pretrained_model: Optional[SklearnClassifierModel] = None


def make_predict(data: list, model_: SklearnClassifierModel) -> List[ModelResponse]:
    predictions = model_.predict(data)
    return [
        ModelResponse(predict=prediction)
        for prediction in predictions
    ]


@app_.get('/')
async def main():
    return 'Welcome to the model hub!'


@app_.post('/predict', response_model=List[ModelResponse])
async def predict(request: XInput):
    logger.debug(f'{request.data=}')
    result = make_predict(request.data, pretrained_model)
    logger.info(f'successful predictions: {result}')
    return result


@app_.get("/health")
def status() -> int:
    model_status = pretrained_model is not None
    logger.info(f"Model is{' not ' if not model_status else ' '}ready")
    if model_status:
        return 200


@app_.on_event('startup')
def get_model():
    global pretrained_model
    logger.info(f"Loading model from 'data/model.joblib'")
    pretrained_model = load_model('data/model.joblib')


if __name__ == '__main__':
    uvicorn.run('app:app_', host='127.0.0.1', port=8000)



