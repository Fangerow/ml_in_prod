import pandas as pd
from unittest import TestCase
from ml_project.raw_data.data_utils import create_data


class TestData(TestCase):
    def test_data_shape(self):
        data = create_data()
        self.assertEqual(data.shape, (297, 14))

    def test_data_type(self):
        data = create_data()
        self.assertTrue(isinstance(data, pd.DataFrame))
