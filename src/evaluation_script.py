import os
import itertools

from more_itertools import one
import datetime

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
        super().__init__(random_state=42)
class RandomForestClassifier_Personalized(RandomForestClassifier):
    def __init__(self):
        self.is_algorithm = True
        super().__init__(random_state=42)
class DecisionTreeRegressor_Personalized(DecisionTreeRegressor):
    def __init__(self):
        self.is_algorithm = True
        super().__init__(random_state=42)
class RandomForestRegressor_Personalized(RandomForestRegressor):
    def __init__(self):
        self.is_algorithm = True
        super().__init__(random_state=42)

class Evaluation_Algorithms():
    def __init__(self, result_folder, dataset, logger):
        self.result_folder = result_folder
        self.dataset_name = dataset.name
        self.dataset_path = dataset.dataset_path
        self.dataset_machinelearning_task = dataset.machinelearning_task
        assert self.dataset_machinelearning_task in ['classification', 'regression']
        self.machinelearning_metrics = None
        self.starting_time = None
        self.finishing_time = None
        self.logger = logger
        self.dataset_characteristics_file_path = None
        self.evaluation_results_file_path = None

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

    def get_algorithms(self):
        '''
        Create tuples of algorithms used in the benchmarking.
        '''
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
        objects = [pipeline_class() for pipeline_class in pipeline]
        algorithm_object = one(filter(lambda obj: hasattr(obj, 'is_algorithm'), objects))
        preprocessing_objects = [obj for obj in objects if not hasattr(obj, 'is_algorithm')]

        self.logger.info(f'Algorithm used: {algorithm_object.__class__.__name__}')
        self.logger.info(f'Data Preprocessing Methods used: {[obj.__class__.__name__ for obj in preprocessing_objects if not obj.__class__.__name__ == 'DummyTransformer']}')
        
        return preprocessing_objects, algorithm_object

    def perform_preprocessing(self, preprocessing_objects, X, y, test_size=0.3):
        '''
        Performs the preprocessing by splitting the dataset into train test split and using a default test size of 0.3, which may be altered later on.
        We need to look at sampling specifically, because it doesn't use the fit_transform method.
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        # Necessary preprocessing
        necessary_preprocessing = make_pipeline(*(RemoveConstColumn(), RemoveDuplicateRows()))
        X_train = necessary_preprocessing.fit_transform(X_train)
        # Perform preprocessing of non-sampling methods using fit_transform
        non_samplers = [obj for obj in preprocessing_objects if not hasattr(obj, 'is_sampler')]
        non_sampler_pipelines = make_pipeline(*non_samplers)
        X_train = non_sampler_pipelines.fit_transform(X_train)
        X_test = non_sampler_pipelines.transform(X_test)
        #Perform preprocessing of sampling methods using fit_resample
        sampler = [obj for obj in preprocessing_objects if hasattr(obj, 'is_sampler')]
        if len(sampler) != 0:
            X_train, y_train = one(sampler).fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test 

    def evaluate_algorithm(self, algorithm_model, X_train, X_test, y_train, y_test):
        '''
        Evaluates the ML algorithm used by evaluating the discrepancy between real test target samples and predicted ones by the algorithm
        '''
        algorithm_model.fit(X_train, y_train)
        y_pred = algorithm_model.predict(X_test)
        self.calculate_scores(y_test, y_pred)

    def calculate_scores(self, y_test, y_pred):
        '''
        Writes the achieved performance scores of the ML algorithm to the machinelearning_metrics attribute
        '''
        if self.dataset_machinelearning_task == 'classification':
            self.machinelearning_metrics = [
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred, average='macro', zero_division=0),
            precision_score(y_test, y_pred, average='macro', zero_division=0),
            recall_score(y_test, y_pred, average='macro', zero_division=0)               
            ]
        if self.dataset_machinelearning_task == 'regression':
            self.machinelearning_metrics = [
            mean_squared_error(y_test, y_pred),
            mean_absolute_error(y_test, y_pred),
            max_error(y_test, y_pred),
            r2_score(y_test, y_pred)         
            ]

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

    def append_result(self, records, preprocessing_objects, algorithm_object):
        '''
        Create new entry in the records list using the preprocessing methods and algorithm used as well as the calculated metrics.
        '''
        if self.dataset_machinelearning_task == 'classification':
            machinelearning_metrics_names = ['Accuracy', 'F1', 'Precision', 'Recall']
        if self.dataset_machinelearning_task == 'regression':
            machinelearning_metrics_names = ['MSE', 'MAE', 'Max_E', 'R2']  
        records.append({
            'Dataset Name':self.dataset_name,
            'Starting Time':self.starting_time,
            'Finishing Time':self.finishing_time,
            **{f'Preprocessing_Method_{i}' : '' if preprocessing_obj.__class__.__name__ == 'DummyTransformer' else preprocessing_obj for i, preprocessing_obj in enumerate(preprocessing_objects)},
            'Algorithm':algorithm_object.__class__.__name__,
            **dict(zip(machinelearning_metrics_names, self.machinelearning_metrics))
        })

    def run(self):
        '''
        Executes Benchmarking.
        '''
        self.logger.info('Start Evaluation')

        self.create_output_datapaths()
        X, y = self.get_dataset()
        algorithms_for_benchmarking  = self.get_algorithms() 
        self.calculate_combinations(X, algorithms_for_benchmarking)

        records = list()
        for idx, combination in enumerate(self.preprocessing_product,1):

            self.logger.info(f'Evaluating combination: {idx}/{len(self.preprocessing_product)}')

            preprocessing_objects, algorithm_object = self.get_preprocessing_algorithm_objects(combination)
            self.starting_time = datetime.datetime.now()
            X_train, X_test, y_train, y_test = self.perform_preprocessing(preprocessing_objects, X, y)
            self.evaluate_algorithm(algorithm_object, X_train, X_test, y_train, y_test)
            self.finishing_time = datetime.datetime.now()
            self.append_result(records,preprocessing_objects,algorithm_object)
        pd.DataFrame(records).to_csv(self.evaluation_results_file_path)

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





