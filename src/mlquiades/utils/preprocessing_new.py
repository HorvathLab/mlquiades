import gzip
import numpy as np
import pandas as pd
import pyranges as pr
from sh import gunzip
import matplotlib.pyplot as plt

def read_in_data_new(
        data_dir, ccle_file, drug_file, ic50_cutoff_value, genes_gtf):
    '''
    Reads in the cancer cell line (ccle) file, the drug response file (the sample data
    provides palbociclib), the ic50 cutoff value (e.g. palbociclib's ic50 value
    is 4), and the genes.gtf file. Keeps cell lines that are overlapping between
    the ccle file and drug response file. Labs the features using the genes.gtf file.
    Prepares dataframe for proper processing in the processing function.
    '''

    if 'gz' in ccle_file:
        gunzip(data_dir + ccle_file)
        ccle_file = ccle_file.replace('.gz', '')
    counts_mtx = pd.read_table(data_dir + ccle_file, header=None, sep=',')
    counts_mtx.iloc[0,0]='Cell line'
    counts_mtx = counts_mtx.T
    counts_mtx.columns=counts_mtx.iloc[0]
    counts_mtx=counts_mtx.drop([0])

    if 'gz' in genes_gtf:
        gunzip(data_dir + genes_gtf)
        genes_gtf = genes_gtf.replace('.gz','')
    gene_def = pr.read_gtf(data_dir + genes_gtf)
    gene_def_df = gene_def.df

    ls = gene_def_df['gene_id'].unique().tolist()
    ls = [x.split('.')[0] for x in ls]
    ls.append('Cell line')
    ls.sort()
    overlapping_genes = list(set(ls) & set(counts_mtx.columns))

    counts_mtx = counts_mtx.loc[counts_mtx['Cell line']!='transcript_ids']
    counts_mtx_subset = counts_mtx[overlapping_genes]

    cell_lines = counts_mtx_subset['Cell line']
    counts_mtx_subset = counts_mtx_subset.drop(columns=['Cell line'])

    gene_def_df = gene_def.df
    gene_def_df = gene_def_df[gene_def_df['transcript_type']=='protein_coding']
    gene_def_df = gene_def_df[['gene_id','gene_name']]

    dictionary_index = [x.split('.')[0] for x in gene_def_df.gene_id]
    gene_dictionary = pd.Series(gene_def_df.gene_name.values, index=dictionary_index)
    gene_dictionary = gene_dictionary.to_dict()

    counts_mtx_subset = counts_mtx_subset[[x for x in \
                                            counts_mtx_subset.columns.tolist() if x in gene_dictionary]]
    new_col_names = [gene_dictionary[x].lower() for x in \
                        counts_mtx_subset.columns.tolist() if x in gene_dictionary]
    counts_mtx_subset.columns = new_col_names

    drug_data = pd.read_table(data_dir + drug_file, sep=',')
    drug_data['Cell line'] = [x.replace('-','').lower() for x in \
                                drug_data['Cell line'].tolist()]

    counts_mtx_subset['Cell line']=[x.split('_')[0].lower() for x in \
                                    cell_lines.tolist()]
    counts_mtx_subset.columns = counts_mtx_subset.columns.str.lower()

    drug_data = drug_data.rename(columns={'Cell line':'cell line'})
    merged_df = pd.merge(drug_data, counts_mtx_subset, on='cell line', how='inner')
    merged_df['label'] = 1
    merged_df['label'][merged_df['IC50']<=ic50_cutoff_value]=-1
    y_labels = merged_df['label']
    metadata = merged_df[['label', 'IC50', 'cell line', 'Tissue']]
    merged_df = merged_df.drop(columns=['label', 'IC50', 'cell line', 'Tissue'])

    merged_df_counts = merged_df.astype('float64')
    merged_df_grouped = merged_df_counts.groupby(by=merged_df_counts.columns, axis=1).mean()

    merged_df_grouped.columns = [col.upper() for col in merged_df_grouped.columns]
    merged_df_grouped = pd.concat([metadata, merged_df_grouped], axis=1)

    return merged_df_grouped, y_labels