import logging
import pandas as pd
from os.path import exists
from typing import Union
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

SklearnClassifierModel = Union[LogisticRegression, KNeighborsClassifier]


def load_model(jbl_path: str) -> SklearnClassifierModel:
    """
    :param jbl_path: path to model weights in joblib format
    :return: pretrained classification model
    """
    if exists(jbl_path):
        model = load(jbl_path)
    else:
        logging.error(f"cant file model weights in {jbl_path}")
        raise FileExistsError
    return model


def load_data(data_path: str) -> pd.DataFrame:
    """
    :param data_path: path to csv data file
    :return: pandas DataFrame with data from data path
    """
    try:
        table_data = pd.read_csv(data_path)
        data = table_data.drop('thalach', axis=1)
        return data
    except FileExistsError as e:
        logging.critical(f'an error: {e}')
        raise FileExistsError
