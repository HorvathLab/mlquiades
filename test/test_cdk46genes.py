import os, sys
import numpy as np
import pandas as pd

dir_ = os.path.dirname(os.path.abspath('src/mlquiades/utils/preprocessing.py'))
sys.path.append(dir_)

from processing import cdk4_6_genes

def test_cdk46genes():
    data_dir = 'sample_data/'
    file_ = 'cdk4_6_genes.txt'
    genes = pd.read_csv(data_dir + file_, header=None).iloc[:,0].to_list()
    genes.append('a')
    df = pd.DataFrame(np.ones((4, len(genes))), columns = genes)
    A, B = cdk4_6_genes(data_dir, df, file_)
    assert A.shape[1] == df.shape[1] - 1
    assert 'B' in locals()

