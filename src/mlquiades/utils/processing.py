import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def pearson(
        X_train_, X_val_, X_test, y_train_):
    '''
    Performs Pearson correlation between every feature and the y-label values
    in the training data only. Maintains features that have a rho value of greater
    than or equal to .3.
    '''
    f = X_train_.apply(lambda c: stats.pearsonr(c.to_numpy(),y_train_.to_numpy().\
            reshape((y_train_.shape[0],))), axis=0)
    f=f.transpose()
    f.columns = ['rho','pvalue']
    f = pd.DataFrame(f).sort_values(by='rho', ascending=False)
    genes = f.index[f.rho>=.3]
    X_train_ = X_train_.loc[:, X_train_.columns.isin(['label', 'IC50', 'cell line', 'Tissue'] + genes)]
    X_val_ = X_val_.loc[:, X_val_.columns.isin(['label', 'IC50', 'cell line', 'Tissue'] + genes)]
    X_test = X_test.loc[:, X_test.columns.isin(['label', 'IC50', 'cell line', 'Tissue'] + genes)]

    return X_train_, y_train_, X_val_, X_test

def cdk4_6_genes(
        data_dir, df, genes_file):
    '''
    Reads in file containing cdk4- and cdk6-related genes. Maintains only
    features that are in that list.
    '''
    genes = pd.read_csv(data_dir + genes_file, header=None).iloc[:,0].to_list()
    x = pd.DataFrame([x.split('_')[0] for x in df.columns])
    x = df.columns[x.isin(['label', 'IC50', 'cell line', 'Tissue'] + genes)[0]]
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
    df_, genes = cdk4_6_genes(data_dir, df, cdk4_6_genes_filename)
    df_genes = pd.DataFrame(genes + cancer_genes.tolist())
    genes = df_genes.iloc[:,0].unique().tolist()
    x = pd.DataFrame([x.split('_')[0] for x in df_.columns])
    x = df.columns[x.isin(['label', 'IC50', 'cell line', 'Tissue'] + genes)[0]]
    df_ = df_.loc[:, x]
    
    return df

def plot_split(df, output_dir):
    '''
    Plot the split for training, validation and testing data with respect to the
    number of sensitive and resistant cancer cell lines in each tissue type.
    '''
    
    fig,ax = plt.subplots(3,1, figsize=(15,10))
    
    p1 = sns.barplot(ax=ax[0], data=df[df['train_val_test']=='train'], x='tissue', 
                     y='value', hue='variable', palette=sns.color_palette('icefire'))
    p1.bar_label(p1.containers[0])
    p1.bar_label(p1.containers[1])
    p1.set_title('TRAIN')
    p1.set_xlabel('')
    p1.set_ylim(top=df['value'].max()+5)
    p1.set_ylabel('# of cell lines')
    p1.set_xticklabels(p1.get_xticklabels(), rotation=45)
    
    p2 = sns.barplot(ax=ax[1], data=df[df['train_val_test']=='val'], x='tissue',
                     y='value', hue='variable', palette=sns.color_palette('icefire'))
    p2.bar_label(p2.containers[0])
    p2.bar_label(p2.containers[1])
    p2.set_title('VAL')
    p2.set_xlabel('')
    p2.set_ylim(top=df['value'].max()+5)
    p2.set_ylabel('# of cell lines')
    p2.set_xticklabels(p2.get_xticklabels(), rotation=45)
    
    p3 = sns.barplot(ax=ax[2], data=df[df['train_val_test']=='test'], x='tissue',
                     y='value', hue='variable', palette=sns.color_palette('icefire'))
    p3.bar_label(p3.containers[0])
    p3.bar_label(p3.containers[1])
    p3.set_title('TEST')
    p3.set_xlabel('tissue type')
    p3.set_ylim(top=df['value'].max()+5)
    p3.set_ylabel('# of cell lines')
    p3.set_xticklabels(p3.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir + '/all_tissues/data_split.png')
    
    return

def split_scale_data(
        data_dir, output_dir, df, y_labels, feature_selection, ros=True, 
        cdk4_6_genes_filename=None, cancer_genes_filename=None):
    '''
    Performs feature selection (3 options) and splits data into training,
    validation and testing datasets. [Splitting intentionally includes
    data from both classes (-1 and 1). Since there is a class imbalance
    and 100 of the samples are sensitive, the split is 60-20-20 for the
    samples in that class. The remaining samples are also split 60-20-20
    within each tissue type.] Scales and normalizes the training,
    validation and testing datasets. Performs random oversampling on the
    training dataset only.
    '''
    if feature_selection == 'cdk4_6_genes':
        df, genes = cdk4_6_genes(data_dir, df, cdk4_6_genes_filename)
    elif feature_selection == 'cdk4_6_cancer_genes':
        df = cdk4_6_cancer_genes(data_dir=data_dir, df=df,
                                 cdk4_6_genes_filename=cdk4_6_genes_filename,
                                 cancer_genes_filename=cancer_genes_filename)

    # isolate the data that pertains to sensitive
    df_sensitive = df[df['label']==-1]
    X_train_sensitive, X_valtest_sensitive, y_train_sensitive, y_valtest_sensitive = train_test_split(
        df_sensitive, df_sensitive['label'], test_size=.4)
    X_val_sensitive, X_test_sensitive, y_val_sensitive, y_test_sensitive = train_test_split(
        X_valtest_sensitive, y_valtest_sensitive, test_size=.5)
    
    X_train_resistant = pd.DataFrame()
    X_val_resistant = pd.DataFrame()
    X_test_resistant = pd.DataFrame()
    y_train_resistant = pd.DataFrame()
    y_val_resistant = pd.DataFrame()
    y_test_resistant = pd.DataFrame()
    df_resistant = df[df['label']==1]
    
    for tissue in df_resistant['Tissue'].unique():
        df_resistant_tissue = df_resistant[df_resistant['Tissue']==tissue]
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
    
    X_train_ = pd.concat([X_train_sensitive, X_train_resistant])    
    X_val_ = pd.concat([X_val_sensitive, X_val_resistant])
    X_test = pd.concat([X_test_sensitive, X_test_resistant])
    y_train_ = pd.concat([y_train_sensitive, y_train_resistant])
    y_val_ = pd.concat([y_val_sensitive, y_val_resistant])
    y_test = pd.concat([y_test_sensitive, y_test_resistant])
        
    metadata_train = X_train_[['label', 'IC50', 'cell line', 'Tissue']]
    metadata_val = X_val_[['label', 'IC50', 'cell line', 'Tissue']]
    metadata_test = X_test[['label', 'IC50', 'cell line', 'Tissue']]
    metadata_train['train_val_test'] = 'train'
    metadata_val['train_val_test'] = 'val'
    metadata_test['train_val_test'] = 'test'
    X_train_ = X_train_.drop(columns=['label', 'IC50', 'cell line', 'Tissue'])
    X_val_ = X_val_.drop(columns=['label', 'IC50', 'cell line', 'Tissue'])
    X_test = X_test.drop(columns=['label', 'IC50', 'cell line', 'Tissue'])
    
    metadata = pd.concat([metadata_train, metadata_val, metadata_test])
    
    if feature_selection == 'pearson':
        X_train_, y_train_, X_val_, X_test = pearson(X_train_, X_val_, X_test,
                                                     y_train_)
    
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)
    X_val_ = scaler.transform(X_val_)
    X_test = scaler.transform(X_test)
    
    grouped = []
    for type in ['train', 'val', 'test']:
        for tissue in metadata['Tissue'].unique():
            meta_sub = metadata[metadata['Tissue']==tissue]
            meta_sub = meta_sub[meta_sub['train_val_test']==type]
            grouped.append([type, tissue, meta_sub['label'].tolist().count(-1),meta_sub['label'].tolist().count(1)])
    grouped = pd.DataFrame(grouped, columns=['train_val_test', 'tissue', 'sensitive', 'resistant'])
    grouped.to_csv(output_dir + '/all_tissues/data_split.csv', index=False)
    grouped = grouped.melt(id_vars=['train_val_test', 'tissue'], value_vars=['sensitive', 'resistant'])
    grouped = grouped.sort('tissue', ascending=False)
    
    plot_split(grouped, output_dir)
    
    if ros:
        ros = RandomOverSampler(random_state=0)
        X_train_, y_train_ = ros.fit_resample(X_train_, y_train_)

    return X_train_, y_train_, X_val_, y_val_, X_test, y_test, metadata