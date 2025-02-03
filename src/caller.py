import logging
import os 
import datetime
import logger_personalized
from evaluation_script import Evaluation_Algorithms, DatasetModel


if __name__ == '__main__':

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_folder = os.path.join('..', 'results')
    current_results_folder = os.path.join(results_folder, f'results_{now}')
    if os.path.isdir(current_results_folder): pass
    else: os.makedirs(current_results_folder)

    logger = logger_personalized.create_logger(logging.DEBUG,current_results_folder)

    benchmark_test = Evaluation_Algorithms(
        current_results_folder,
        DatasetModel('sfgd', os.path.join('..','Datasets','kc1.csv'), 'classification'),
        logger
    )
    benchmark_test.run_benchmarking()