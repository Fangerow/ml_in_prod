import numpy
from unittest import TestCase
from ml_project.scr.transformer.transformer import transformer_pipeline
from fit_predict_pipeline_test import generate_synt_data
from ml_project.raw_data.data_utils import create_data
from sklearn.utils.validation import check_is_fitted


class TestCustomTransformer(TestCase):
    def test_model(self):
        data = create_data()
        fake_data = generate_synt_data(data)
        fake_design_matrix = fake_data.drop('condition', axis=1)
        model = transformer_pipeline(data)
        predictions = model.predict(fake_design_matrix)

        self.assertIsNone(check_is_fitted(model))
        self.assertEqual(predictions.shape, (30, ))
        self.assertTrue(isinstance(predictions, numpy.ndarray))
        self.assertTrue(len(numpy.unique(predictions)) > 0)  # test model predict different labels

