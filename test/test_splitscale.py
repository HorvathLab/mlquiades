import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

dir_ = os.path.dirname(os.path.abspath('src/mlquiades/utils/preprocessing.py'))
sys.path.append(dir_)

from processing import cdk4_6_genes, cdk4_6_cancer_genes, pearson, split_scale_data

def test_splitscale():
    '''
    Unit test the random oversampling and the data splitting.
    '''
    fake_data_a = np.concatenate((np.arange(0,10), np.ones(90)), axis=0)
    fake_data_a = np.tile(fake_data_a, (80,1)).transpose()
    fake_data_b = np.arange(0,100)*(-1)
    fake_data_b = np.tile(fake_data_b, (20,1)).transpose()
    fake_data = np.concatenate((fake_data_a, fake_data_b), axis=1)
    df_ = pd.DataFrame(fake_data)
    y_labels = pd.DataFrame(np.concatenate((np.ones(90), np.ones(10) - 1),
        axis=0))
    
    dir_ = 'sample_data/'
    A_1, B_1, C_1, D_1, E_1, F_1 = split_scale_data(data_dir=dir_, df=df_, 
            y_labels=y_labels, feature_selection='pearson', ros=True, 
            cdk4_6_genes_filename=None, cancer_genes_filename=None)
    A_2, B_2, C_2, D_2, E_2, F_2 = split_scale_data(data_dir=dir_, df=df_,
        y_labels=y_labels, feature_selection='pearson', ros=False,
        cdk4_6_genes_filename=None, cancer_genes_filename=None)
    assert A_1.shape[0] + C_1.shape[0] + E_1.shape[0] > df_.shape[0]
    assert A_2.shape[0] + C_2.shape[0] + E_2.shape[0] == df_.shape[0]
    assert A_2.shape[0] == 60
    assert C_2.shape[0] == 20
    assert E_2.shape[0] == 20

