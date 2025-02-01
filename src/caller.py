import logging
from evaluation_script import DatasetClass
import os 
import pandas as pd


logging_level = logging.INFO

dataset_models = [
    DatasetClass(
        'sfgd', os.path.join('..','Datasets','kc1.csv'), 'classification'
    )
]



if __name__ == '__main__':
    pass