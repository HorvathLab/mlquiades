import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from src.mlquiades.utils.processing import *

def test_split():
    '''
    Unit test the random oversampling and the data splitting.
    '''

    fake_data_a = np.concatenate((np.arange(0,10), np.ones(90)), axis=0)
    fake_data_a = np.tile(fake_data_a, (80,1)).transpose()
    fake_data_b = np.arange(0,100)*(-1)
    fake_data_b = np.tile(fake_data_b, (20,1)).transpose()
    fake_data = np.concatenate((fake_data_a, fake_data_b), axis=1)
    df_ = pd.DataFrame(fake_data)
    y_labels = pd.DataFrame(np.concatenate((np.ones(90), np.ones(10) - 2),
        axis=0))
    y_labels = y_labels.rename(columns={0:'label'})
    df_ = pd.concat([df_, y_labels], axis=1)
    df_['tissue'] = 'breast'
    df_['tissue'][1:20] = 'liver'
    df_['tissue'][90:93] = 'liver'
    df_['tissue'][21:30]= 'lung'
    df_['tissue'][93:96]= 'lung'
    df_['cell line'] = np.arange(1, df_.shape[0] + 1)
    df_['cell line'] = [str(x) for x in df_['cell line'].tolist()]
    df_['for_pearson_calculation'] = np.random.rand(df_.shape[0], 1)

    dir_ = 'test/'
    A_1, B_1, C_1, D_1, E_1, F_1, G_1, H_1 = split_data(
        output_dir=dir_, df=df_)
    
    assert A_1.shape[0] + C_1.shape[0] + E_1.shape[0] == df_.shape[0]
    assert ['data_split.png' in os.listdir('test/')]