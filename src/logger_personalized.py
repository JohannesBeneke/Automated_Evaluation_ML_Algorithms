import logging
import os

def create_logger(level, result_folder_location):
    '''
    Create logger
    '''
    logger = logging.getLogger('Algorithm Evaluation')
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(os.path.join(result_folder_location,'log.log'))
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger