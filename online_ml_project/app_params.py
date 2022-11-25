import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

SklearnClassifierModel = Union[LogisticRegression, KNeighborsClassifier]


class XInput(BaseModel):
    data: List[List[int]]


class ModelResponse(BaseModel):
    predict: np.float64
