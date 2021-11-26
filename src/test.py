from itertools import combinations_with_replacement
import pandas as pd
import numpy as np
import os
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class DatasetClass():
    def __init__(self, name, dataset_path, machinelearning_task):
        assert os.path.isfile(dataset_path)
        self.name = name
        self.dataset_path = dataset_path
        self.machinelearning_task = machinelearning_task

class Base_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X):
        return self
    def transform(self, X):
        return X


class Benchmark_Algorithms():
    # def __init__(self, result_folder, dataset):
    def __init__(self, dataset):
        # self.result_folder = result_folder
        self.name = dataset.name
        self.dataset_path = dataset.dataset_path
        self.machinelearning_task = dataset.machinelearning_task
        assert self.machinelearning_task in ['classification', 'regression']


    def calculate_combinations_for_evaluation(self):
        '''
        Calculate the combinations of data preprocessing methods and machine learning algorithms to evaluate in the benchmarking.
        The resulting combinations are calculated by using a factorial design that enables the calculation of every possible combination.
        '''
        combinations=[
            (Base_Transformer, StandardScaler),
            (Base_Transformer, PCA),
            (DecisionTreeClassifier, RandomForestClassifier),
        ]

        calculated_combinations = list(itertools.product(*combinations))
        return calculated_combinations

    def run(self):
        
        combinations = self.calculate_combinations_for_evaluation()
        print(combinations)

if __name__ == '__main__':

    # now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # results_folder = os.path.join('..', 'results')
    # current_results_folder = os.path.join(results_folder, f'results_{now}')
    # if os.path.isdir(current_results_folder): pass
    # else: os.makedirs(current_results_folder)
    
    
    # dataset_model = DatasetClass('sfgd', os.path.join('..','Datasets', 'kc1.csv'), 'classification')

    benchmark_test = Benchmark_Algorithms(
        DatasetClass('sfgd', os.path.join('..','Datasets', 'kc1.csv'), 'classification')
    )
    benchmark_test.run()






