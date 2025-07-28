import os
import sys
from sh import gunzip

dir_ = os.path.dirname(os.path.abspath('src/mlquiades/utils/preprocessing.py'))
sys.path.append(dir_)
from preprocessing import read_in_data

def test_readin():
    '''
    Unit test confirming the read_in_data function produces a
    data frame variable and a y_labels variable that has a maximum
    value of 1.
    '''

    data_dir = 'sample_data/'
    files = os.listdir('sample_data/')
    ccle_file = [x for x in files if 'CCLE' in x][0]
    drug_file = 'palbociclib.csv'
    genes_gtf = [x for x in files if 'gencode' in x][0]

    if 'gz' in ccle_file:
        gunzip(data_dir + ccle_file)
    ccle_file = ccle_file.replace('.gz', '')

    if 'gz' in genes_gtf:
        gunzip(data_dir + genes_gtf)
    genes_gtf = genes_gtf.replace('.gz', '')
    
    ic50_cutoff_value = 4
    merged_df_grouped, y_labels = read_in_data(data_dir, ccle_file, drug_file,
            ic50_cutoff_value, genes_gtf)
    
    assert max(y_labels) == 1
    assert 'merged_df_grouped' in locals()
