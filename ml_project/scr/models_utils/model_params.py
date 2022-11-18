from dataclasses import dataclass


@dataclass()
class ModelParams:
    model_name: str
    params: dict
