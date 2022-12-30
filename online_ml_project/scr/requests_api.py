import requests
import logging

from .logger import initialize_logger
from .utils import load_data
from app import get_model

cfg = {
    'data_path': "data/heart_cleveland_upload.csv",
    'localhost': '0.0.0.0',
    'port': 8000,
    'endpoint': 'predict'
}
logger = logging.getLogger(__name__)


def make_pred(cfg: dict, logger_=logger):
    get_model()
    url_prefix = f"{cfg['localhost']}:{cfg['port']}"
    get_model()
    logger = initialize_logger(logger_)

    logger.info("Reading data")
    data = load_data(cfg['data_path']).drop("condition", axis=1)

    request_data = data.to_numpy().tolist()

    logger.info("Sending post request")
    response = requests.post(url=f"http://{url_prefix}/{cfg['endpoint']}",
                             json={'data': request_data}
                             )
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response data samples:\n {response.json()}")


if __name__ == "__main__":
    make_pred(cfg, logger)
