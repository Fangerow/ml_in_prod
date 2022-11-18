import logging

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from typing import Union
from ml_project.logger import initialize_logger

logger = logging.getLogger(__name__)
logger = initialize_logger(logger)
SklearnClassifierModel = Union[LogisticRegression, KNeighborsClassifier]


def predict(pretrained_model: SklearnClassifierModel,
            test_sample: pd.DataFrame):
    """
    :param pretrained_model:  pretrained sklearn knn or logistic regression scr
    :param test_sample: features (vectors), that should be classified
    """
    predicted_labels = pretrained_model.predict(test_sample)
    logger.debug('successful predictions')
    return predicted_labels

