from itertools import combinations_with_replacement
import os
import itertools

from more_itertools import one
import datetime
import logging 

from preprocessing_methods import *

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score

class DatasetModel():
    '''
    Class that enables the creation of a dataset object including the name, the file path and the corresponding machine learning algorithm
    '''
    def __init__(self, name, dataset_path, machinelearning_task):
        assert os.path.isfile(dataset_path)
        self.name = name
        self.dataset_path = dataset_path
        self.machinelearning_task = machinelearning_task

# To add further algorithms, add here a class with the is_algorithm attribute to differentiate algorithms from preprocessing methods
class DecisionTreeClassifier_Personalized(DecisionTreeClassifier):
    def __init__(self):
        self.is_algorithm = True
        super().__init__()
class RandomForestClassifier_Personalized(RandomForestClassifier):
    def __init__(self):
        self.is_algorithm = True
        super().__init__()
class DecisionTreeRegressor_Personalized(DecisionTreeRegressor):
    def __init__(self):
        self.is_algorithm = True
        super().__init__()
class RandomForestRegressor_Personalized(RandomForestRegressor):
    def __init__(self):
        self.is_algorithm = True
        super().__init__()

class Evaluation_Algorithms():
    def __init__(self, result_folder, dataset, logger):
        self.result_folder = result_folder
        self.dataset_name = dataset.name
        self.dataset_path = dataset.dataset_path
        self.dataset_machinelearning_task = dataset.machinelearning_task
        assert self.dataset_machinelearning_task in ['classification', 'regression']
        self.calculated_combinations = None
        self.machinelearning_metrics = None
        self.number_preprocessing_stages = None
        self.number_preprocessing_combinations = None
        self.starting_time = None
        self.finishing_time = None
        self.logger = logger
        self.dataset_characteristics_file_path = None
        self.evaluation_results_file_path = None
        self.results_dataframe = None

    def get_dataset(self):
        '''
        Read dataset from the dataset path and split the column of the target variable from the rest of the dataset
        '''
        dataset = pd.read_csv(self.dataset_path)
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
        return X, y

    def calculate_combinations(self, data_to_evaluate, algorithms):
        '''
        Calculate the combinations of data preprocessing methods and machine learning algorithms to evaluate in the benchmarking.
        The resulting combinations are calculated by using a factorial design that enables the calculation of every possible combination.
        '''
        # Check for data quality metrics to skip some methods not applicable to the dataset.
        # Check for missing values and categorical data in dataset since some preprocessing methods and algorithm cannot work with those.
        if data_to_evaluate.isna().sum().sum() == 0:
            imputation_methods = (DummyTransformer,)
        else:
            imputation_methods = (MeanImputer, MedianImputer,MostFrequentImputer)
        
        if len(data_to_evaluate.columns[data_to_evaluate.dtypes == 'object']) == 0:
            encoding_methods = (DummyTransformer, )
        else:
            encoding_methods = (TargetEncoder,OrdinalEncoder,OneHotEncoder)

        if self.dataset_machinelearning_task == 'classification':
            sampling_methods = (DummyTransformer, OverSampling, UnderSampling, CombineSampling)
        elif self.dataset_machinelearning_task == 'regression':
            sampling_methods = (DummyTransformer, )
        
        neccessary_preprocessing = (RemoveConstColumn, RemoveDuplicateRows)

        # Calculate the cartesian product of all preprocessing methods in order to evaluate all possible combinations
        preprocessing_methods = [
            imputation_methods,
            encoding_methods,
            (DummyTransformer, StandardScaling, MinMaxScaling),
            (DummyTransformer, PCA_New),
            sampling_methods,
            algorithms
        ]

        self.preprocessing_product = list(itertools.product(*preprocessing_methods))
        self.number_preprocessing_combinations = len(self.preprocessing_product)
        self.number_preprocessing_stages = len(preprocessing_methods)
        self.logger.info(f'Number of Preprocessing Stages: {self.number_preprocessing_stages}')

    def get_algorithms(self):
        if self.dataset_machinelearning_task == 'classification':
            algorithms = (
                DecisionTreeClassifier_Personalized,
                RandomForestClassifier_Personalized
            )
        elif self.dataset_machinelearning_task == 'regression':
            algorithms = (
                DecisionTreeRegressor_Personalized,
                RandomForestRegressor_Personalized
            )
        return algorithms

    def get_preprocessing_algorithm_objects(self, pipeline):
        '''
        Returns the objects of the preprocessing methods used in this combination in a list as well as the object of the ML algorithm used in this combination
        '''
        class_objects = [pipeline_class() for pipeline_class in pipeline]
        algorithm_object = one([class_object for class_object in class_objects if hasattr(class_object, 'is_algorithm')])
        self.logger.info(f'Algorithm used: {algorithm_object.__class__.__name__}')
        preprocessing_objects = [class_object for class_object in class_objects if not hasattr(class_object, 'is_algorithm')]
        self.logger.info(f'Data Preprocessing Methods used: {[x.__class__.__name__ for x in preprocessing_objects if not x.__class__.__name__ == 'DummyTransformer']}')
        return preprocessing_objects, algorithm_object

    def perform_preprocessing(self, preprocessing_objects, X, y, test_size=0.3):
        '''
        Performs the preprocessing by splitting the dataset into train test split and using a default test size of 0.3, which may be altered later on.
        We need to look at sampling specifically, because it doesn't use the fit_transform method.
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        non_samplers = [preprocessing_obj for preprocessing_obj in preprocessing_objects if not hasattr(preprocessing_obj, 'is_sampler')]
        non_sampler_pipelines = make_pipeline(*non_samplers)
        X_train_preprocessed = non_sampler_pipelines.fit_transform(X_train)
        X_test_preprocessed = non_sampler_pipelines.transform(X_test)

        sampler = [preprocessing_obj for preprocessing_obj in preprocessing_objects if hasattr(preprocessing_obj, 'is_sampler')]
        if len(sampler) != 0:
            X_train_preprocessed, y_train = one(sampler).fit_resample(X_train_preprocessed, y_train)

        return X_train_preprocessed, X_test_preprocessed, y_train, y_test

    def evaluate_algorithm(self, algorithm, X_train, X_test, y_train, y_test):
        '''
        Evaluates the ML algorithm used by evaluating the discrepancy between real test target samples and predicted ones by the algorithm
        '''
        algorithm_model = algorithm
        algorithm_model.fit(X_train, y_train)
        y_pred = algorithm_model.predict(X_test)
        self.calculate_scores(y_test, y_pred)

    def calculate_scores(self, y_test, y_pred):
        '''
        Writes the achieved performance scores of the ML algorithm to the machinelearning_metrics attribute
        '''
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

    def create_output_datapaths(self):
        '''
        Creation of the output datapaths in the result folder. When evaluating multiple dataset at once, for every dataset a result folder is created based on the dataset name.
        Inside the result folder for the evaluated dataset the paths for the csv files containing the evaluation results and dataset characteristics are created.
        '''
        self.dataset_results_folder = os.path.join(self.result_folder, self.dataset_name)
        if os.path.isdir(self.dataset_results_folder): pass
        else: os.makedirs(self.dataset_results_folder)
        self.dataset_characteristics_file_path = os.path.join(self.dataset_results_folder, f'{self.dataset_name}_dataset_characteristics.csv')
        self.evaluation_results_file_path = os.path.join(self.dataset_results_folder, f'{self.dataset_name}_evaluation_results.csv')  

    def create_preprocessing_dict(self, preprocessing_objects):
        number_preprocessing_methods = ['Preprocessing_Method_'+str(count) for count in range(self.number_preprocessing_stages)]
        return dict(zip(number_preprocessing_methods, [x.__class__.__name__ for x in preprocessing_objects if x.__class__.__name__ != 'DummyTransformer']))
    
    def create_results_dict(self, algorithm):
        algorithm_dict = {'Algorithm': algorithm}
        if self.dataset_machinelearning_task == 'classification':
            score_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        elif self.dataset_machinelearning_task == 'regression':
            score_names = ['MSE', 'MAE', 'Max_E', 'R2']
        scores_dict = dict(zip(score_names, self.machinelearning_metrics))
        return algorithm_dict, scores_dict

    def add_to_results(self, preprocessing_objects, algorithm_object):
        preprocessing_dict = self.create_preprocessing_dict(preprocessing_objects)
        algorithm_dict, scores_dict = self.create_results_dict(algorithm_object)
        result_dict = {
            'Name':self.dataset_name,
            'Starting Time':self.starting_time,
            'Finishing Time':self.finishing_time,
            **preprocessing_dict,
            **algorithm_dict,
            **scores_dict
        }
        return result_dict

    def create_results_dataframe(self):
        self.results_dataframe = pd.DataFrame()

    def create_output_file(self, preprocessing_objects, algorithm_object):
        result_dict = self.add_to_results(preprocessing_objects, algorithm_object)
        if self.dataset_machinelearning_task == 'classification':
            evaluation_results = pd.DataFrame(
                [result_dict]
            )
        
        evaluation_results.to_csv(self.evaluation_results_file_path)
        #return evaluation_results

    # def write_results_to_file(self):


    def run_benchmarking(self):
        self.logger.info('Start Evaluation')
        self.create_output_datapaths()
        algorithms_for_benchmarking  = self.get_algorithms()
        
        X_evaluation, y_evaluation = self.get_dataset()
        self.calculate_combinations(X_evaluation, algorithms_for_benchmarking)

        number_of_combinations = self.number_preprocessing_combinations
        for combination in self.preprocessing_product:
            self.logger.info(f'Number of combinations to solve: {number_of_combinations}')
            preprocessing_objects, algorithm_object = self.get_preprocessing_algorithm_objects(combination)
            self.starting_time = datetime.datetime.now()
            X_train, X_test, y_train, y_test = self.perform_preprocessing(preprocessing_objects, X_evaluation, y_evaluation)
            self.evaluate_algorithm(algorithm_object, X_train, X_test, y_train, y_test)
            self.finishing_time = datetime.datetime.now()
            self.logger.info(f'ML Scores: {self.machinelearning_metrics}')
            self.create_output_file(preprocessing_objects, algorithm_object)
            number_of_combinations -= 1
        self.create_output_datapaths()

if __name__ == '__main__':

    pass
    # now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # results_folder = os.path.join('..', 'results')
    # current_results_folder = os.path.join(results_folder, f'results_{now}')
    # if os.path.isdir(current_results_folder): pass
    # else: os.makedirs(current_results_folder)

    # logger = create_logger(logging.DEBUG)

    # benchmark_test = Evaluation_Algorithms(
    #     current_results_folder,
    #     DatasetModel('sfgd', os.path.join('..','Datasets','kc1.csv'), 'classification')
    # )
    # benchmark_test.run_preprocessing()





