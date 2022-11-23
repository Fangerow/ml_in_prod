from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

SklearnClassifierModel = Union[LogisticRegression, KNeighborsClassifier]


@dataclass()
class XInput(BaseModel):
    data: List[Union[float, str]]


@dataclass()
class ModelResponse(BaseModel):
    predict: int
