import logging


def initialize_logger(logger: logging.Logger) -> logging.Logger:
    """
    :return: current module logger
    """
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(1)
    return logger
