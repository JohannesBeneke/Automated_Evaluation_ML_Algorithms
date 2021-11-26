from itertools import combinations_with_replacement
import pandas as pd
import numpy as np
import os
import itertools
import datetime

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score

class DatasetClass():
    '''
    Class that enables the creation of a dataset object including the name, the file path and the corresponding machine learning algorithm
    '''
    def __init__(self, name, dataset_path, machinelearning_task):
        assert os.path.isfile(dataset_path)
        self.name = name
        self.dataset_path = dataset_path
        self.machinelearning_task = machinelearning_task

class Base_Transformer(BaseEstimator, TransformerMixin):
    '''
    Basic Transformer that is used to enable combinations of data preprocessing methods without the use of a specific method 
    '''
    def __init__(self):
        pass
    def fit(self, X):
        return self
    def transform(self, X):
        return X

class DecisionTreeClassifier_Personalized(DecisionTreeClassifier):
    def __init__(self):
        self.is_algorithm = True
        super().__init__()

class RandomForestClassifier_Personalized(RandomForestClassifier):
    def __init__(self):
        self.is_algorithm = True
        super().__init__()


class Evaluation_Algorithms():
    def __init__(self, result_folder, dataset):
        self.result_folder = result_folder
        self.dataset_name = dataset.name
        self.dataset_path = dataset.dataset_path
        self.dataset_machinelearning_task = dataset.machinelearning_task
        assert self.dataset_machinelearning_task in ['classification', 'regression']
        self.calculated_combinations = None
        self.machinelearning_metrics = None

    def get_dataset(self):
        dataset = pd.read_csv(self.dataset_path)
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
        return X, y

    def calculate_combinations_for_evaluation(self):
        '''
        Calculate the combinations of data preprocessing methods and machine learning algorithms to evaluate in the benchmarking.
        The resulting combinations are calculated by using a factorial design that enables the calculation of every possible combination.
        '''
        combinations=[
            (Base_Transformer, StandardScaler),
            (Base_Transformer, PCA),
            (DecisionTreeClassifier_Personalized, RandomForestClassifier_Personalized),
        ]
        self.calculated_combinations = list(itertools.product(*combinations))

    def get_preprocessing_algorithm_objects(self, pipeline):
        class_objects = [pipeline_class() for pipeline_class in pipeline]
        algorithm_object = [class_object for class_object in class_objects if hasattr(class_object, 'is_algorithm')]
        preprocessing_objects = [class_object for class_object in class_objects if not hasattr(class_object, 'is_algorithm')]
        return preprocessing_objects, *algorithm_object

    def perform_preprocessing(self, preprocessing_objects, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        preprocessing_pipelines = make_pipeline(*preprocessing_objects)
        X_train_preprocessed = preprocessing_pipelines.fit_transform(X_train)
        X_test_preprocessed = preprocessing_pipelines.transform(X_test)
        return X_train_preprocessed, X_test_preprocessed, y_train, y_test

    def calculate_scores(self, y_test, y_pred):
        if self.dataset_machinelearning_task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            self.machinelearning_metrics = [accuracy, f1, precision, recall]
        if self.dataset_machinelearning_task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            max_e = max_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.machinelearning_metrics = [mse, mae, max_e, r2]

    def evaluate_algorithm(self, algorithm, X_train, X_test, y_train, y_test):
        algorithm_model = algorithm
        algorithm_model.fit(X_train, y_train)
        y_pred = algorithm_model.predict(X_test)
        self.calculate_scores(y_test, y_pred)

    def set_output_files(self):
        self.dataset_results_folder = os.path.join(self.result_folder, self.dataset_name)
        if os.path.isdir(self.dataset_results_folder): pass
        else: os.makedirs(self.dataset_results_folder)


    def run(self):
        
        X_evaluation, y_evaluation = self.get_dataset()
        self.calculate_combinations_for_evaluation()
        for combination in self.calculated_combinations:
            preprocessing_objects, algorithm_object = self.get_preprocessing_algorithm_objects(combination)
            X_train, X_test, y_train, y_test = self.perform_preprocessing(preprocessing_objects, X_evaluation, y_evaluation)
            self.evaluate_algorithm(algorithm_object, X_train, X_test, y_train, y_test)
            print(self.machinelearning_metrics)
        self.set_output_files()

if __name__ == '__main__':

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_folder = os.path.join('..', 'results')
    current_results_folder = os.path.join(results_folder, f'results_{now}')
    if os.path.isdir(current_results_folder): pass
    else: os.makedirs(current_results_folder)

    benchmark_test = Evaluation_Algorithms(
        current_results_folder,
        DatasetClass('sfgd', os.path.join('..','Datasets', 'kc1.csv'), 'classification')
    )
    benchmark_test.run()






