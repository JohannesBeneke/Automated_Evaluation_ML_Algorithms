import pandas as pd
import numpy as np

# Import transformer class from sklearn in order to fit and transform data. In order for Transformermixin to work we also need BaseEstimator
from sklearn.base import BaseEstimator, TransformerMixin

## Cleaning methods
# Missing values
from sklearn.impute import SimpleImputer

## Transformation
# Encoding
import category_encoders as ce
# Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Reduction
# Dimensionality Reduction
from sklearn.decomposition import PCA

## Balancing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


# Essentials
class RemoveConstColumn(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return X.loc[:, (X != X.iloc[0]).any()]

class RemoveDuplicateRows(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return X.drop_duplicates(keep='first')

# Missing values
class DummyImputer(TransformerMixin):
    '''
    Create Dummyclass for imputing to reuse code. Impute numerical columns with specific strategy while imputing columns of dtype object with the most frequent value
    '''
    def __init__(self, strategy):
        self.strategy = strategy
    def fit(self, X, y=None): 
        self.series = pd.Series(
            [X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else getattr(X[c], self.strategy)() for c in X],
            index = X.columns
            )
        return self
    def transform(self, X, y=None):
        return X.fillna(self.series)

class MeanImputer(DummyImputer):
    def __init__(self):
        super().__init__(strategy='mean')
class MedianImputer(DummyImputer):
    def __init__(self):
        super().__init__(strategy='median')
class MostFrequentImputer(SimpleImputer):
    def __init__(self):
        super().__init__(strategy = "most_frequent")


# Encoder
class TargetEncoder(ce.TargetEncoder):
    def __init__(self):
        super().__init__(return_df=True, handle_missing='return_nan')
class OrdinalEncoder(ce.OrdinalEncoder):
    def __init__(self):
        super().__init__(return_df=True, handle_missing='return_nan')
class OneHotEncoder(ce.OneHotEncoder):
    def __init__(self):
        super().__init__(return_df=True, handle_missing='return_nan')

# Scaling
class StandardScaling(StandardScaler):
    def __init__(self):
        super().__init__()
class MinMaxScaling(MinMaxScaler):
    def __init__(self):
        super().__init__()

# Dimension Reduction
class PCA_New(PCA):
    def __init__(self):
        super().__init__(n_components=0.95, random_state=42)

# Balancing
class OverSampling(SMOTE):
    def __init__(self):
        super().__init__(random_state=42)
class UnderSampling(RandomUnderSampler):
    def __init__(self):
        super().__init__(random_state=42)
class CombineSampling(SMOTEENN):
    def __init__(self):
        super().__init__(random_state=42)

if __name__ == "__main__":
    import os
    sfgd = pd.read_csv(os.path.join('..','Datasets', 'kc1.csv'))
    #X = sfgd.iloc[:,:-1]
    
    transformer = MostFrequentImputer()
    X = pd.DataFrame(data={'col1':[0,1,6, 2,1, 5, 900, np.nan], 'col2':['A', 'B', 'A','C','D', 'B', 'A',np.nan]})
    print(X)
    X = transformer.fit_transform(X)
    print(X)
