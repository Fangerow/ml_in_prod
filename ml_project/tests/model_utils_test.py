from unittest import TestCase

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from ml_project.scr.models_utils.train import create_model, train_model
from sklearn.utils.validation import check_is_fitted
from ml_project.scr.models_utils.predict import predict
from ml_project.scr.models_utils.model_params import ModelParams


class TestUtils(TestCase):
    def test_model_creation(self):
        params_ = ModelParams(model_name='logistic_regression', params={})
        self.assertTrue(isinstance(create_model(params_), LogisticRegression))

        params_ = ModelParams(model_name='k_neighbor', params={})
        self.assertTrue(isinstance(create_model(params_), KNeighborsClassifier))

        with self.assertRaises(ValueError):
            params_ = ModelParams(model_name='efficientnet_b7', params={})
            _ = create_model(params_)

    def test_train(self):
        params_ = ModelParams(model_name='k_neighbor', params={'n_neighbors': 1})
        model = train_model(params_, pd.DataFrame([[0] * 14]), pd.DataFrame([0]))
        self.assertIsNone(check_is_fitted(model))

    def test_predict(self):
        model = LogisticRegression()  # In this test we can consider it as pretrained
        model.fit(pd.DataFrame([[0] * 14, [1] * 14]), pd.DataFrame([0, 1]))
        fake_data = pd.DataFrame([[0] * 14])
        self.assertEqual(predict(model, fake_data), 0)
