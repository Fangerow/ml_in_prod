import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf


def parse_yaml(yaml_file_path: str, hydra=True):

    """
    :param hydra: bool: is true if hydra is implemented
    :param yaml_file_path: path to YAML config
    :return: config data in easy dict format
    """
    if hydra:
        data = yaml.safe_load(OmegaConf.to_yaml(yaml_file_path))
        return edict(data)
    else:
        with open(yaml_file_path, 'r') as file:
            data = edict(yaml.safe_load(file))
            return data
