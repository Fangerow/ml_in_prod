import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from typing import Union
from ml_project.logger import initialize_logger
from .model_params import ModelParams


logger = logging.getLogger(__name__)
logger = initialize_logger(logger)
SklearnClassifierModel = Union[LogisticRegression, KNeighborsClassifier]


def create_model(model_params: ModelParams)\
        -> SklearnClassifierModel:
    """
    :param model_name: name of the scr to be created
    :param model_params: scr parameters according to sklearn documentation
    :return: Sklearn Classifier Model
    """
    if model_params.model_name == 'logistic_regression':
        model = LogisticRegression(**model_params.params)
    elif model_params.model_name == 'k_neighbor':
        model = KNeighborsClassifier(**model_params.params)
    else:
        logger.critical(f'scr {model_params.model_name} is not supported, you can use only knn or log_reg')
        raise ValueError

    return model


def train_model(model_params: ModelParams,
                design_matrix,
                labels):
    """
    :param model_name: name of the scr to be trained
    :param model_params: model initial params according to it's documentation
    :param design_matrix: feature vectorized matrix
    :param labels:
    :return: pre-trained Sklearn Classifier Model
    """
    model = create_model(model_params)
    pipe = make_pipeline(StandardScaler(), model)  # thanks to EDA we know, that the data is unbalanced
    pipe.fit(design_matrix, labels)
    logger.info(f'model has achieved training score {pipe.score(design_matrix, labels)}')
    return pipe
