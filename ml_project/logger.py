import logging


def initialize_logger(logger: logging.Logger, level=1) -> logging.Logger:
    """
    :return: current module logger
    """
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(level)
    return logger
