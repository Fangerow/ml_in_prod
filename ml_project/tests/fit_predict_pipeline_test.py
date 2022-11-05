from unittest import TestCase
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
import logging
from collections import namedtuple
from sklearn.utils.validation import check_is_fitted
from ml_project.raw_data.data_utils import create_data
from ml_project.scr.model_training_pipeline import Learner
from ml_project.logger import initialize_logger
from ml_project.scr.models_utils.model_params import ModelParams

Column = namedtuple('Column', 'mean std min max nunic type')


def generate_synt_data(real_data: pd.DataFrame) -> pd.DataFrame:
    """
    :param real_data: reference data to create synthetic
    :return: Dataframe with synthetic data, that is near to real
    """
    data = real_data
    column_params = []
    for column in data:  # to get data template params
        cur_column = data[column]
        column_params.append(Column(cur_column.mean(),
                                    cur_column.std(),
                                    cur_column.min(),
                                    cur_column.max(),
                                    cur_column.nunique(),
                                    cur_column.dtype))

    table = []
    num_vec = 30
    for params in column_params:  # data generation according to template
        mu, sigma = params.mean, params.std
        generated_vector = (np.random.normal(mu, sigma, num_vec) if params.nunic > 5
                            else np.random.randint(params.max + 1, size=num_vec))
        # to generate fake data in referenced data range (min, max)
        new_column = list(map(lambda x: max(params.min, x % (params.max + 1)), generated_vector))
        table.append(np.array(new_column, dtype=params.type))
    table = np.array(table).T
    synt_data = pd.DataFrame(table, columns=data.columns.values.tolist())
    return synt_data


class EducationTest(TestCase):
    def test_learner_module(self):
        data = create_data()
        fake_data = generate_synt_data(data)
        self.assertEqual(fake_data.shape, (30, 14))

        logger = logging.getLogger(__name__)
        logger = initialize_logger(logger)
        self.assertTrue(isinstance(logger, logging.Logger))  # test for logger.py function

        learning = Learner(logger, fake_data, 3)
        design_matrix, labels = learning.data_preprocessing(drop_trash_feature=False)
        self.assertEqual(design_matrix.shape, (30, 13))
        self.assertEqual(labels.shape, (30,))

        params = {
            'max_iter': 120,
            'C': 0.9,
        }
        with self.assertRaises(ValueError):
            params_ = ModelParams(model_name='CatBoos', params=params)
            learning.train_and_dump(params_)

        fake_predictios = np.random.randint(2, size=30)
        self.assertIsNotNone(learning.compute_metric(recall_score, labels, fake_predictios))
        self.assertLess(learning.compute_metric(recall_score, labels, fake_predictios), 1)

    def test_education(self):
        logger = logging.getLogger(__name__)
        logger = initialize_logger(logger)

        data = create_data()
        fake_data = generate_synt_data(data)
        learning = Learner(logger, fake_data, 3)
        params = {
            'max_iter': 120,
            'C': 0.9,
        }
        params_ = ModelParams(model_name='logistic_regression', params=params)
        model = learning.train_and_dump(params_,
                                        save_weights=False)
        self.assertIsNone(check_is_fitted(model))
