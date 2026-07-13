import os
import sys
import numpy as np
import pandas as pd

from src.mlquiades.utils.processing import *

def test_cdk46cancer():
    '''
    Unit test that ensures the number of columns in the dataframe for
    the cdk4_6 feature selection is less than or equal to that of the 
    cdk4_6_cancer feature selection.
    '''

    data_dir = 'sample_data/'
    file_cdk46 = 'cdk4_6_genes.txt'
    file_cancer = 'cancer_genes.tsv'
    
    genes = pd.read_csv(data_dir + file_cdk46, header=None).iloc[:,0].to_list()
    genes.append('a')
    genes = [x.lower() for x in genes]
    df = pd.DataFrame(np.ones((4, len(genes))), columns = genes)

    A, B = cdk4_6_genes(data_dir, df, file_cdk46)
    C = cdk4_6_cancer_genes(data_dir, df, file_cdk46, file_cancer)
    
    assert A.shape[1] <= C.shape[1]
