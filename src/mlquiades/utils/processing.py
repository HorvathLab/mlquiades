import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from . import *

def pearson(
        X_train_, X_val_, X_test, pearson_train):
    '''
    Performs Pearson correlation between every feature and the y-label values
    in the training data only. Maintains features that have a rho value of greater
    than or equal to .3.
    '''
    f = X_train_.apply(lambda c: stats.pearsonr(c.to_numpy(), pearson_train.to_numpy().\
            reshape((pearson_train.shape[0],))), axis=0)
    f=f.transpose()
    f.columns = ['rho','pvalue']
    f = pd.DataFrame(f).sort_values(by='rho', ascending=False)
    genes = f.index[f.rho>=.3]

    X_train_ = X_train_.loc[:, X_train_.columns.isin(genes)]
    X_val_ = X_val_.loc[:, X_val_.columns.isin(genes)]
    X_test = X_test.loc[:, X_test.columns.isin(genes)]

    return X_train_, X_val_, X_test

def cdk4_6_genes(
        data_dir, df, genes_file):
    '''
    Reads in file containing cdk4- and cdk6-related genes. Maintains only
    features that are in that list.
    '''
    genes = pd.read_csv(data_dir + genes_file, header=None).iloc[:,0].to_list()
    genes = [x.lower() for x in genes]
    x = pd.DataFrame([x.split('_')[0] for x in df.columns])
    x = df.columns[x.isin(['label', 'cell line', 'tissue'] + genes)[0]]
    df = df.loc[:, x]

    return df, genes

def cdk4_6_cancer_genes(
        data_dir, df, cdk4_6_genes_filename, cancer_genes_filename):
    '''
    Reads in files containing cdk4-, cdk6-, and cancer-related genes. Maintains
    only features that are in that list.
    '''
    cancer_genes = pd.read_csv(data_dir + cancer_genes_filename, sep='\t')
    cancer_genes = cancer_genes['Gene Symbol'].unique()
    cancer_genes = [x.lower() for x in cancer_genes]
    df_, genes = cdk4_6_genes(data_dir, df, cdk4_6_genes_filename)
    df_genes = pd.DataFrame(genes + cancer_genes)
    genes = df_genes.iloc[:, 0].unique().tolist()
    x = pd.DataFrame([x.split('_')[0] for x in df.columns])
    x = df.columns[x.isin(['label', 'cell line', 'tissue'] + genes)[0]]
    df = df.loc[:, x]
    
    return df

def split_data(
        output_dir, df):
    '''
    Splits data into training, validation and testing datasets. [Splitting
    intentionally includes data from both classes (-1 and 1). Since there is
    a class imbalance and 100 of the samples are sensitive, the split is
    60-20-20 for the samples in that class. The remaining samples are also
    split 60-20-20 within each tissue type.]
    '''

    # isolate the data that pertains to the sensitive class
    df_sensitive = df[df['label']==-1]
    breast_sensitive = df_sensitive[df_sensitive['tissue']=='breast']
    df_sensitive = df_sensitive[df_sensitive['tissue']!='breast']
    X_train_sensitive, X_valtest_sensitive, y_train_sensitive, y_valtest_sensitive = train_test_split(
        df_sensitive, df_sensitive['label'], test_size=.4)
    X_val_sensitive, X_test_sensitive, y_val_sensitive, y_test_sensitive = train_test_split(
        X_valtest_sensitive, y_valtest_sensitive, test_size=.5)
    
    # isolate the data that pertains to the resistant class
    X_train_resistant = pd.DataFrame()
    X_val_resistant = pd.DataFrame()
    X_test_resistant = pd.DataFrame()
    y_train_resistant = pd.DataFrame()
    y_val_resistant = pd.DataFrame()
    y_test_resistant = pd.DataFrame()
    leftover_resistant = pd.DataFrame()
    df_resistant = df[df['label']==1]
    breast_resistant = df_resistant[df_resistant['tissue']=='breast']
    df_resistant = df_resistant[df_resistant['tissue']!='breast']
    
    # breast cancer samples - split so that you have both classes in train, val and test (if possible)
    X_train_sensitive_b = pd.DataFrame()
    X_val_sensitive_b = pd.DataFrame()
    X_test_sensitive_b = pd.DataFrame()
    y_train_sensitive_b = pd.DataFrame()
    y_val_sensitive_b = pd.DataFrame()
    y_test_sensitive_b = pd.DataFrame()
    X_train_resistant_b = pd.DataFrame()
    X_val_resistant_b = pd.DataFrame()
    X_test_resistant_b = pd.DataFrame()
    y_train_resistant_b = pd.DataFrame()
    y_val_resistant_b = pd.DataFrame()
    y_test_resistant_b = pd.DataFrame()
    
    if len(breast_sensitive)>0:
        X_train_sensitive_b, X_valtest_sensitive_b, y_train_sensitive_b, y_valtest_sensitive_b = train_test_split(
            breast_sensitive, breast_sensitive['label'], test_size=.4)
        X_val_sensitive_b, X_test_sensitive_b, y_val_sensitive_b, y_test_sensitive_b = train_test_split(
            X_valtest_sensitive_b, y_valtest_sensitive_b, test_size=.5)
    if len(breast_resistant)>0:
        X_train_resistant_b, X_valtest_resistant_b, y_train_resistant_b, y_valtest_resistant_b = train_test_split(
            breast_resistant, breast_resistant['label'], test_size=.4)
        X_val_resistant_b, X_test_resistant_b, y_val_resistant_b, y_test_resistant_b = train_test_split(
            X_valtest_resistant_b, y_valtest_resistant_b, test_size=.5)
    
    for tissue in df_resistant['tissue'].unique():
        df_resistant_tissue = df_resistant[df_resistant['tissue']==tissue]
        if df_resistant_tissue.shape[0]>3:
            X_train_tissue, X_valtest_tissue, y_train_tissue, y_valtest_tissue = train_test_split(
                df_resistant_tissue, df_resistant_tissue['label'], test_size=.4)
            X_val_tissue, X_test_tissue, y_val_tissue, y_test_tissue = train_test_split(
                X_valtest_tissue, y_valtest_tissue, test_size=.5)
            X_train_resistant = pd.concat([X_train_resistant, X_train_tissue])
            X_val_resistant = pd.concat([X_val_resistant, X_val_tissue])
            X_test_resistant = pd.concat([X_test_resistant, X_test_tissue])
            y_train_resistant = pd.concat([y_train_resistant, y_train_tissue])
            y_val_resistant = pd.concat([y_val_resistant, y_val_tissue])
            y_test_resistant = pd.concat([y_test_resistant, y_test_tissue])
        else:
            leftover_resistant = pd.concat([leftover_resistant, df_resistant_tissue])
    if leftover_resistant.shape[0]>0:
        X_train_leftover, X_valtest_leftover, y_train_leftover, y_valtest_leftover = train_test_split(
            leftover_resistant, leftover_resistant['label'], test_size=.4)
        X_val_leftover, X_test_leftover, y_val_leftover, y_test_leftover = train_test_split(
            X_valtest_leftover, y_valtest_leftover, test_size=.5)

    if leftover_resistant.shape[0]<1:
        X_train_ = pd.concat([X_train_sensitive, X_train_resistant, X_train_resistant_b, X_train_sensitive_b])
        X_val_ = pd.concat([X_val_sensitive, X_val_resistant, X_val_sensitive_b, X_val_resistant_b])
        X_test = pd.concat([X_test_sensitive, X_test_resistant, X_test_sensitive_b, X_test_resistant_b])
        y_train_ = pd.concat([y_train_sensitive, y_train_resistant, y_train_sensitive_b, y_train_resistant_b])
        y_val_ = pd.concat([y_val_sensitive, y_val_resistant, y_val_sensitive_b, y_val_resistant_b])
        y_test = pd.concat([y_test_sensitive, y_test_resistant, y_test_sensitive_b, y_test_resistant_b])
    else:
        X_train_ = pd.concat([X_train_sensitive, X_train_resistant, X_train_resistant_b, X_train_sensitive_b,
                          X_train_leftover])
        X_val_ = pd.concat([X_val_sensitive, X_val_resistant, X_val_sensitive_b, X_val_resistant_b,
                        X_val_leftover])
        X_test = pd.concat([X_test_sensitive, X_test_resistant, X_test_sensitive_b, X_test_resistant_b,
                        X_test_leftover])
        y_train_ = pd.concat([y_train_sensitive, y_train_resistant, y_train_sensitive_b, y_train_resistant_b,
                          y_train_leftover])
        y_val_ = pd.concat([y_val_sensitive, y_val_resistant, y_val_sensitive_b, y_val_resistant_b,
                        y_val_leftover])
        y_test = pd.concat([y_test_sensitive, y_test_resistant, y_test_sensitive_b, y_test_resistant_b,
                        y_test_leftover])

    metadata_train = X_train_[['label', 'cell line', 'tissue']]
    metadata_val = X_val_[['label', 'cell line', 'tissue']]
    metadata_test = X_test[['label', 'cell line', 'tissue']]
    metadata_train['train_val_test'] = 'train'
    metadata_val['train_val_test'] = 'val'
    metadata_test['train_val_test'] = 'test'
    pearson_train = X_train_['for_pearson_calculation']
    X_train_ = X_train_.drop(columns=['label', 'cell line', 'tissue', 'for_pearson_calculation'])
    X_val_ = X_val_.drop(columns=['label', 'cell line', 'tissue', 'for_pearson_calculation'])
    X_test = X_test.drop(columns=['label', 'cell line', 'tissue', 'for_pearson_calculation'])
    
    metadata = pd.concat([metadata_train, metadata_val, metadata_test])
    grouped = []
    for type in ['train', 'val', 'test']:
        for tissue in metadata['tissue'].unique():
            meta_sub = metadata[metadata['tissue']==tissue]
            meta_sub = meta_sub[meta_sub['train_val_test']==type]
            grouped.append([type, tissue, ','.join(meta_sub['cell line'].tolist()),
                            meta_sub['label'].tolist().count(-1),
                            meta_sub['label'].tolist().count(1)])

    grouped = pd.DataFrame(grouped, columns=['train_val_test', 'tissue', 'cell lines', 'sensitive', 
                                             'resistant'])
    grouped.to_csv(output_dir + '/data_split.csv', index=False)
    grouped = grouped.melt(id_vars=['train_val_test', 'tissue'], value_vars=['sensitive',
                                                                             'resistant'])
    grouped = grouped.sort_values(by='tissue', ascending=True)
    
    all_tissues_df = []
    for type in ['train', 'val', 'test']:
        meta_sub = metadata[metadata['train_val_test']==type]
        all_tissues_df.append([type, 'all_tissues', meta_sub['label'].tolist().count(-1),
                            meta_sub['label'].tolist().count(1)])
    all_tissues_df = pd.DataFrame(all_tissues_df, columns=['train_val_test', 'tissue',
                                                               'sensitive', 'resistant'])
    all_tissues_df = all_tissues_df.melt(id_vars=['train_val_test', 'tissue'], 
                                         value_vars=['sensitive', 'resistant'])
    grouped = pd.concat([all_tissues_df, grouped])
    
    plot_split(grouped, output_dir)
    
    return X_train_, y_train_, X_val_, y_val_, X_test, y_test, pearson_train, metadata


def select_features(
    data_dir, X_train_, X_val_, X_test, pearson_train, feature_selection, 
    cdk4_6_genes_filename=None, cancer_genes_filename=None):
    '''
    Performs feature selection.
    '''
    
    if feature_selection == 'cdk4_6_genes':
        X_train_, genes = cdk4_6_genes(data_dir, X_train_, cdk4_6_genes_filename)
        X_val_, genes = cdk4_6_genes(data_dir, X_val_, cdk4_6_genes_filename)
        X_test, genes = cdk4_6_genes(data_dir, X_test, cdk4_6_genes_filename)
    elif feature_selection == 'cdk4_6_cancer':
        X_train_ = cdk4_6_cancer_genes(
            data_dir=data_dir, df=X_train_,
            cdk4_6_genes_filename=cdk4_6_genes_filename,
            cancer_genes_filename=cancer_genes_filename)
        X_val_ = cdk4_6_cancer_genes(
            data_dir=data_dir, df=X_val_,
            cdk4_6_genes_filename=cdk4_6_genes_filename,
            cancer_genes_filename=cancer_genes_filename)
        X_test = cdk4_6_cancer_genes(
            data_dir=data_dir, df=X_test,
            cdk4_6_genes_filename=cdk4_6_genes_filename,
            cancer_genes_filename=cancer_genes_filename)
    elif feature_selection == 'pearson':
        X_train_, X_val_, X_test = pearson(X_train_, X_val_, X_test, pearson_train)
    
    return X_train_, X_val_, X_test

def scale_and_transform(X_train_, X_val_, X_test):
    '''
    Scales and normalizes the training,
    validation and testing datasets.
    '''
    
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)
    X_val_ = scaler.transform(X_val_)
    X_test = scaler.transform(X_test)
    
    return X_train_, X_val_, X_test

def ros_run(X_train_, y_train_):
    '''
    Performs random oversampling on the training dataset only.
    '''
    
    ros = RandomOverSampler(random_state=0)
    X_train_, y_train_ = ros.fit_resample(X_train_, y_train_)
    return X_train_, y_train_
