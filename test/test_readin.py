import os
import sys

dir_ = os.path.dirname(os.path.abspath('src/mlquiades/utils/preprocessing.py'))
sys.path.append(dir_)
from preprocessing import read_in_data

def test_readin():
    data_dir = 'sample_data/'
    ccle_file = 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz'
    drug_file = 'palbociclib.csv'
    genes_gtf = 'gencode.v19.genes.v7_model.patched_contigs.gtf.gz'
    ic50_cutoff_value = 4
    merged_df_grouped, y_labels = read_in_data(data_dir, ccle_file, drug_file,
            ic50_cutoff_value, genes_gtf)
    assert max(y_labels) == 1
    assert 'merged_df_grouped' in locals()
