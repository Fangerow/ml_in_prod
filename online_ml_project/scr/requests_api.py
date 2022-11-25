import requests
import json
import logging

from logger import initialize_logger
from utils import load_data


PATH_TO_DATA = "data/heart_cleveland_upload.csv"
LOCALHOST = '127.0.0.1'
PORT = 15000
DOMAIN = f"{LOCALHOST}:{PORT}"
ENDPOINT = '/predict'
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger = initialize_logger(logger)

    logger.info("Reading data")
    data = load_data(PATH_TO_DATA).drop("condition", axis=1)

    request_data = data.to_numpy().tolist()
    logger.info(f"Request data samples:\n {request_data[::5]}")

    logger.info("Sending post request")
    response = requests.post(url=f"http://{DOMAIN}/{ENDPOINT}",
                             json={'data': request_data}
                             )
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response data samples:\n {response.json()[::5]}")
