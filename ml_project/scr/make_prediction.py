import logging
import click
import csv
import pandas as pd
from os.path import exists
from typing import Union
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from ml_project.logger import initialize_logger
from .models_utils.predict import predict

SklearnClassifierModel = Union[LogisticRegression, KNeighborsClassifier]
logger = logging.getLogger(__name__)
logger = initialize_logger(logger)


def load_model(jbl_path: str) -> SklearnClassifierModel:
    """
    :param jbl_path: path to model weights in joblib format
    :return: pretrained classification model
    """
    if exists(jbl_path):
        model = load(jbl_path)
        logger.info("model successfully loaded from joblib file")
    else:
        logger.error(f"cant file model weights in {jbl_path}")
        raise FileExistsError
    return model


def load_data(data_path: str) -> pd.DataFrame:
    """
    :param data_path: path to csv data file
    :return: pandas DataFrame with data from data path
    """
    try:
        table_data = pd.read_csv(data_path)
        return table_data
    except FileExistsError as e:
        logger.critical(f'an error: {e}')
        raise FileExistsError


def save_data(desired_path: str, predictions):
    """
    :param desired_path: path to save the predicted result
    :param predictions: your model predictions
    """
    with open(desired_path, "w") as pred_store_csv:
        csv_writer = csv.writer(pred_store_csv, delimiter=";")
        for label in predictions:
            csv_writer.writerow([label])


def make_predictions(model: SklearnClassifierModel,
                     data: pd.DataFrame,
                     path: str
                     ):
    logger.info("Started predict pipeline.")
    labels = predict(model, data)
    save_data(path, labels)
    logger.info(f"predictions loaded to {path}")


@click.command()
@click.argument('model_path')
@click.argument('data_csv_path')
@click.argument('path')
def run_predict(model_path: str,
                data_csv_path: str,
                path: str):
    data = load_data(data_csv_path)
    model = load_model(model_path)
    make_predictions(model, data, path)


if __name__ == "__main__":
    run_predict()
