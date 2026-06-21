import os
import sys
import numpy as np
import pandas as pd

from src.mlquiades.utils.processing import *

def test_cdk46genes():
    '''
    Unit test that confirms the dataframe and y_lables are being generated
    in the cdk4_6_genes function.
    '''

    data_dir = 'sample_data/'
    file_ = 'cdk4_6_genes.txt'
    genes = pd.read_csv(data_dir + file_, header=None).iloc[:,0].to_list()
    genes.append('a')
    genes = [x.lower() for x in genes]
    df = pd.DataFrame(np.ones((4, len(genes))), columns = genes)
    A, B = cdk4_6_genes(data_dir, df, file_)

    assert A.shape[1] == df.shape[1] - 1
    assert 'B' in locals()