import os
import sys
import string
import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from src.mlquiades.utils.processing import *

def test_pearson():
    '''
    Unit test that the pearson function from processing.py
    calculates correlation properly across features and returns
    a dataframe with features that have a rho value of >=.3. It
    does this by generating synthetic data.
    '''

    fake_data_a = np.arange(0,100)
    fake_data_a = np.tile(fake_data_a, (80,1)).transpose()
    fake_data_b = np.arange(0,100)*(-1)
    fake_data_b = np.tile(fake_data_b, (20,1)).transpose()
    fake_data = np.concatenate((fake_data_a, fake_data_b), axis=1)
    X_train_ = pd.DataFrame(fake_data)
    X_val_ = X_train_
    X_test = X_train_
    y_train_ = pd.DataFrame(np.arange(0,100))
        
    colnames = ['a']
    for i in range(100):
        RL1 = random.choice(string.ascii_letters)
        RL2 = random.choice(string.ascii_letters)
        RL3 = random.choice(string.ascii_letters)
        colnames.append(RL1+RL2+RL3)
    
    colnames = list(set(colnames))[0:100]
    X_train_.columns = colnames
    X_val_.columns = colnames
    X_test.columns = colnames
    A, B, C = pearson(X_train_, X_val_, X_test, y_train_)
    
    assert A.shape[1] == 80
