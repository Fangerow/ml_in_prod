from easydict import EasyDict as edict
from dataclasses import dataclass


@dataclass()
class TrainingPipelineParams:
    cfg: edict

