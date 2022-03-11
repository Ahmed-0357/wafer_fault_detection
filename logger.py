import logging
import os


def set_logger(logger, dir_name, file_name):
    """set logger level, formate and file handler 

    Args:
        logger (class): logging class
        dir_name (str): directory name where the log file will be located
        file_name (str): log file name

    Returns:
        class: logging class
    """
    
    logger.setLevel(logging.DEBUG)  # set level

    if not os.path.exists(dir_name):  # directory check
        os.makedirs(dir_name)

    file_handler = logging.FileHandler(
        os.path.join(dir_name, file_name))
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s:%(funcName)s:%(lineno)s:%(message)s')
    file_handler.setFormatter(formatter)  # set fromate
    logger.addHandler(file_handler)  # set file handler
    
    return logger
    