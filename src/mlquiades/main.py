import os
import sys
import argparse
from utils import *

print('....... Parsing arguments .......')

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--a',
        type=str,
        action='store',
        dest='data_dir', 
        help='data directory (required)')
    parser.add_argument(
        '--b', 
        type=str,
        action='store',
        dest='output_folder_name',
        default='output',
        help='output folder name (optional)')
    parser.add_argument(
        '--d',
        type=bool,
        action='store',
        dest='confusion', 
        default=False,
        help='plot confusion matrices per tissue type')
    parser.add_argument(
        '--f',
        type=bool,
        action='store',
        dest='ros', 
        default=True,
        help='randomly oversample training data (optional)')
    parser.add_argument(
        '--g',
        type=int,
        action='store',
        dest='step_size_nodes',
        default=5,
        help='step size for nodes in hyperparameter tuning for nn w/ hyperband'
        '(optional)')
    parser.add_argument(
        '--i',
        type=int,
        action='store',
        dest='min_nodes',
        default=5,
        help='minimum number of nodes in hyperparameter tuning for nn w/ hyperband'
        '(optional)')
    parser.add_argument(
        '--j',
        type=int,
        action='store',
        dest='max_nodes',
        default=512,
        help='maximum number of nodes in hyperparameter tuning for nn w/ hyperband'
        '(optional)')
    parser.add_argument(
        '--k',
        type=int,
        action='store',
        dest='max_trials',
        default=10,
        help='maximum number of trials in hyperparameter tuning for nn w/ hyperband'
        '(optional)')
    parser.add_argument(
        '--l',
        type=int,
        action='store',
        dest='executions_per_trial',
        default=3,
        help='number of executions per trial in hyperparameter tuning for nn w/ hyperband (optional)')
    parser.add_argument(
        '--m',
        type=int,
        action='store',
        dest='patience',
        default=3,
        help='patience in early stopping for nn w/ hyperband (optional)')
    parser.add_argument(
        '--n',
        type=float,
        action='store',
        dest='min_delta',
        default=.01,
        help='minimum delta in early stopping for nn w/ hyperband (optional)')
    parser.add_argument(
        '--o',
        type=int,
        action='store',
        dest='epochs',
        default=100,
        help='epochs in hyperparameter tuning for nn w/ hyperband (optional)')
    parser.add_argument(
        '--p',
        type=float,
        action='store',
        dest='learning_rate_min',
        default=1e-4,
        help='learning rate minimum in hyperparameter tuning for nn w/ hyperband'
        '(optional)')
    parser.add_argument(
        '--q',
        type=float,
        action='store',
        dest='learning_rate_max',
        default=1e-2,
        help='learning rate maximum in hyperparameter tuning for nn w/ hyperband'
        '(optional)')
    parser.add_argument(
        '--r',
        type=int,
        action='store',
        dest='max_layers',
        default=20,
        help='the maximum number of layers for the neural net hyperband parameter tuning (optional)')
    parser.add_argument(
        '--s',
        type=str,
        action='store',
        dest='confusion',
        default=False,
        help='plot confusion matrices for individual tissue types (optional)')

    return parser

def main():
    parser = params()
    args = parser.parse_args()
    data_dir = args.data_dir + '/'
    output_dir = args.output_folder_name
    confusion = args.confusion
    ros = args.ros
    step_size_nodes = args.step_size_nodes
    min_nodes = args.min_nodes
    max_nodes = args.max_nodes
    max_trials = args.max_trials
    executions_per_trial = args.executions_per_trial
    patience = args.patience
    min_delta = args.min_delta
    epochs = args.epochs
    learning_rate_min = args.learning_rate_min
    learning_rate_max = args.learning_rate_max
    confusion = args.confusion
    max_layers = args.max_layers
    cdk4_6_filename = 'cdk4_6_genes.txt'
    cancer_genes_filename = 'cancer_genes.tsv'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if confusion:
        os.mkdir(output_dir + '/confusion')

    print('....... Reading in data ......................')
    
    df = pd.read_csv(data_dir + '/gex_palbociclib.csv')
    df2 = pd.read_csv(data_dir + '/isoforms_palbociclib.csv')
    df = df.merge(df2, how='inner', on=['cell line', 'ic50', 'auc', 'max_conc', 'label', 'tissue'])
    
    df['for_pearson_calculation'] = df['ic50']
    df = df.dropna(subset=['label']).drop(columns=['ic50', 'auc', 'max_conc'])
    
    print('....... Splitting and scaling data ...........')
    X_train_split, y_train, X_val_split, y_val_, X_test_split, y_test, pearson_train, metadata = split_data(
        output_dir=output_dir, df=df)
    
    feature_selections = ['cdk4_6_genes', 'cdk4_6_cancer', 'pearson']
    datatypes_searchsymbols = [(['gex'], ['ensg']), (['isoforms'], ['enst']), (['both'], ['ensg', 'enst'])]
    
    for data_type, search_symbol in datatypes_searchsymbols:
        for feature_select in feature_selections:
            output_dir_feature = output_dir + '/' + feature_select + '_' + data_type[0]
            X_train_ = X_train_split.iloc[:, X_train_split.columns.str.contains('|'.join(search_symbol))]
            X_val_ = X_val_split.iloc[:, X_val_split.columns.str.contains('|'.join(search_symbol))]
            X_test = X_test_split.iloc[:, X_test_split.columns.str.contains('|'.join(search_symbol))]
            if not os.path.isdir(output_dir_feature):
                os.mkdir(output_dir_feature)
            
            X_train_, X_val_, X_test = select_features(
                data_dir, X_train_, X_val_, X_test, pearson_train, feature_select,
                cdk4_6_genes_filename=cdk4_6_filename, cancer_genes_filename=cancer_genes_filename)
            X_train_, X_val_, X_test = scale_and_transform(X_train_, X_val_, X_test)
            if ros:
                X_train_, y_train_ = ros_run(X_train_, y_train)
            
            print('....... Building and evaluating models .......')
            nn_hb = neural_net_with_hyperband(
                X_train_, y_train_, X_val_, y_val_, X_test, y_test, data_dir,
                step_size_nodes, min_nodes, max_nodes, max_trials, executions_per_trial,
                patience, min_delta, epochs, learning_rate_min, learning_rate_max, max_layers,
                metadata, output_dir=output_dir, plt_confusion=confusion)
            rf = random_forest(
                X_train_, y_train_, X_test, y_test, output_dir_feature, feature_select, metadata,
                output_dir=output_dir, plt_confusion=confusion)
            ridge = ridge_classifier(
                X_train_, y_train_, X_test, y_test, output_dir_feature, feature_select, metadata,
                output_dir=output_dir, plt_confusion=confusion)
            evaluation_df = pd.concat([nn_hb, rf, ridge])
            evaluation_df.columns = ['model', 'tissue', 'acc', 'rocauc', 'n_correctly_predicted_sensitive_cell_lines', \
                'n_correctly_predicted_resistant_cell_lines']
            evaluation_df.to_csv(output_dir_feature + '/evaluation_df.csv', index=False)
            
            print('....... Generating evaluation reports ........')
            plot_combined_rocauc(evaluation_df, feature_select, output_dir_feature)
            plot_combined_acc(evaluation_df, feature_select, output_dir_feature)
        stitch_pngs(output_dir, data_type[0])

if __name__=='__main__': 
    main()
