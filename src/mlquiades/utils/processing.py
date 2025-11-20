import pandas as pd
from scipy import stats
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
    df = df.loc[:, df.columns.isin(['label', 'IC50', 'cell line', 'Tissue'] + genes)]
    
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
    df = df.loc[:, df.columns.isin(['label', 'IC50', 'cell line', 'Tissue'] + genes)]
    
    return df

def split_scale_data(
        data_dir, df, y_labels, feature_selection, ros=True, 
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
    
    # X_train_, X_valtest, y_train_, y_valtest = train_test_split(df, y_labels,
    #                                                             test_size=.4)
    # X_val_, X_test, y_val_, y_test = train_test_split(X_valtest, y_valtest,
    #                                                   test_size=.5)
    
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
    if ros:
        ros = RandomOverSampler(random_state=0)
        X_train_, y_train_ = ros.fit_resample(X_train_, y_train_)

    return X_train_, y_train_, X_val_, y_val_, X_test, y_test, metadata