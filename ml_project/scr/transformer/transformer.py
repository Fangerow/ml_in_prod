import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self.changeable_feature = feature

    def fit(self, x: pd.DataFrame, y=None):
        """
        fit the transformer to input data
        """
        return self

    def transform(self, design_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        transform the data with already fitted transformer
        """
        x_copied = design_matrix.copy()
        x_copied = x_copied.drop(self.changeable_feature, axis=1)
        return x_copied

    def inverse_transform(self, design_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        transorms revertion
        """
        x_copied = design_matrix.copy()
        x_copied[self.changeable_feature] = design_matrix[self.changeable_feature]
        return x_copied


def transformer_pipeline(data: pd.DataFrame):
    """
    call function to create transformer
    """
    pipe = Pipeline(
        steps=[
            ('custom_transformer', CustomTransformer('thalach')),
            ('logreg_model', LogisticRegression())
        ]
    )
    train_x, train_y = data.drop('condition', axis=1), data['condition']
    pipe.fit(train_x, train_y)
    return pipe
