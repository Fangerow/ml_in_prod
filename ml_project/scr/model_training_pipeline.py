import logging

import hydra
import numpy
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_project.configs.parse_yaml import parse_yaml
from ml_project.logger import initialize_logger
from ml_project.raw_data.data_utils import create_data
from ml_project.scr.models_utils.train import train_model, create_model
from .models_utils.model_params import ModelParams
from .model_training_params import TrainingPipelineParams


class Learner:
    def __init__(self, logger_, raw_data, num_folds):
        self.logger = logger_
        self.raw_data = raw_data
        self.k_folds = num_folds

    def data_preprocessing(self, drop_trash_feature=True):
        if drop_trash_feature:
            self.raw_data = self.raw_data.drop('thalach', axis=1)
        x_train, y_train = self.raw_data.values[:, :-1], self.raw_data.values[:, -1]
        self.logger.info(f'design matrix shape is {x_train.shape} and labels shape is {y_train.shape}')
        return x_train, y_train

    def compute_metric(self, metric,
                       tru_val: numpy.array,
                       predictions: numpy.array) -> float:
        result_value = metric(tru_val, predictions)
        self.logger.debug(f'computed {metric} value is {result_value}')
        return result_value

    def train_and_dump(self, model_params: ModelParams,
                       save_weights=True):
        des_matrix, labels = self.data_preprocessing()
        self.logger.info('Model training started')
        trained_model = train_model(model_params, des_matrix,  labels)
        if save_weights:
            dump(trained_model, 'model.joblib')

        # metric estimation
        estimator = create_model(model_params)
        for train_idxs, val_idxs in StratifiedKFold(n_splits=3, shuffle=True).split(des_matrix, labels):
            train_x_, val_x_ = des_matrix[train_idxs], des_matrix[val_idxs]
            train_y_, val_y_ = labels[train_idxs], labels[val_idxs]
            model = estimator
            pipe = make_pipeline(StandardScaler(), model)  # thanks to EDA we know, that the data is unbalanced
            pipe.fit(train_x_, train_y_)
            preds = pipe.predict(val_x_)
            self.logger.debug(f'sklearn f1 score is {self.compute_metric(f1_score, preds, val_y_)}')

        return trained_model


@hydra.main(version_base=None,
            config_path='../configs',
            )
def train_pipeline(cfg: str):
    pipeline_params = parse_yaml(cfg).ppl
    logger = logging.getLogger(__name__)
    logger = initialize_logger(logger)
    logger.info('Training is started..')
    raw_data = create_data()
    logger.info('data ready')

    learning = Learner(logger, raw_data, pipeline_params.split_strategy.num_folds)

    params = pipeline_params.model_params.params
    name_of_model = pipeline_params.model_params.model_name

    params_ = ModelParams(model_name=name_of_model, params=params)
    learning.train_and_dump(params_)


if __name__ == "__main__":
    train_pipeline()
