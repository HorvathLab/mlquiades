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
        '--c',
        type=str,
        action='store',
        dest='data_filename',
        help='data filename (required)')
    parser.add_argument(
        '--cc',
        type=str,
        action='store',
        dest='data_filename_2',
        help='data filename 2 (optional)')
    parser.add_argument(
        '--d',
        type=bool,
        action='store',
        dest='confusion', 
        default=False,
        help='plot confusion matrices per tissue type')
    parser.add_argument(
        '--e',
        type=bool,
        action='store',
        dest='rocauc', 
        default=False,
        help='plot rocauc per tissue type')
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
        help='step size for nodes in hyperparameter tuning for nn w/ hyperband '
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
        type=str,
        action='store',
        dest='data_type',
        default='both',
        help='The data type going into the model. Options include: gex, isoforms, both.')
    parser.add_argument(
        '--s',
        type=str,
        action='store',
        dest='cdk4_6_genes_filename',
        help='filename for cdk4 and cdk6 genes for the feature selection option cdk4_6_genes')
    parser.add_argument(
        '--t',
        type=str,
        action='store',
        dest='cancer_genes_filename',
        help='filename for cancer genes for the feature selection option cdk4_6_cancer_genes')
    parser.add_argument(
        '--u',
        type=str,
        action='store',
        dest='drug',
        help='drug name (e.g. palbociclib or ribociclib)')
    parser.add_argument(
        '--v',
        type=str,
        action='store',
        dest='gdsc',
        help='gdsc version (e.g. gdsc1 or gdsc2)')
    parser.add_argument(
        '--w',
        type=str,
        action='store',
        dest='sensitivity_metric',
        help='drug sensitivity metric (e.g. ic50 or auc)')

    return parser

def main():
    parser = params()
    args = parser.parse_args()
    data_dir = args.data_dir + '/'
    output_dir = args.output_folder_name
    data_filename = args.data_filename
    data_filename_2 = args.data_filename_2
    confusion = args.confusion
    rocauc = args.rocauc
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
    data_type = args.data_type
    cdk4_6_filename = args.cdk4_6_genes_filename
    cancer_genes_filename = args.cancer_genes_filename
    drug = args.drug
    gdsc = args.gdsc
    drug = drug.lower()
    gdsc = gdsc.lower()
    sensitivity_metric = args.sensitivity_metric        

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if data_type not in ['gex', 'isoforms', 'both']:
        raise TypeError('The data_type (option -r) is incorrect. Please supply either gex, isoforms, \
                        or both')
    
    print('....... Reading in data ......................')
    df = pd.read_csv(data_dir + data_filename)
    if data_filename_2 != None:
        df2 = pd.read_csv(data_dir + data_filename_2)
        df = df.merge(df2.iloc[:, 16:], left_index=True, right_index=True, how='inner')
    
    if sensitivity_metric=='ic50':
        df['label'] = df['label_' + drug + '_' + gdsc]
    else:
        df['label'] = df[sensitivity_metric.upper() + '_' + gdsc + '_' + drug]
    columns_label = [x for x in df.columns if 'gdsc' in x]
    df['for_pearson_calculation'] = df[sensitivity_metric.upper() + '_' + gdsc + '_' + drug]
    df = df.dropna(subset=['label_' + drug + '_' + gdsc]).drop(columns=columns_label)
    
    print('....... Splitting and scaling data ...........')
    X_train_split, y_train, X_val_split, y_val_, X_test_split, y_test, pearson_train, metadata = split_data(
        output_dir=output_dir, df=df, drug_response_metric=sensitivity_metric)
    
    feature_selections = ['cdk4_6_genes', 'cdk4_6_cancer', 'pearson']
    
    if data_type == 'both':
        data_types = ['gex', 'isoforms']
    else:
        data_types = data_type
    
    for data_type in data_types:
        if data_type == 'gex':
            search_symbol = 'ensg'
        else:
            search_symbol = 'enst'
        
        for feature_select in feature_selections:
            output_dir_feature = output_dir + '/' + feature_select + '_' + data_type
            X_train_ = X_train_split.iloc[:, X_train_split.columns.str.contains(search_symbol)]
            X_val_ = X_val_split.iloc[:, X_val_split.columns.str.contains(search_symbol)]
            X_test = X_test_split.iloc[:, X_test_split.columns.str.contains(search_symbol)]
            
            if not os.path.isdir(output_dir_feature):
                os.mkdir(output_dir_feature)
            
            X_train_, X_val_, X_test = select_features(
                data_dir, X_train_, X_val_, X_test, pearson_train, feature_select,
                cdk4_6_genes_filename=cdk4_6_filename, cancer_genes_filename=cancer_genes_filename)
            X_train_, X_val_, X_test = scale_and_transform(X_train_, X_val_, X_test)
            X_train_, y_train_ = ros_run(X_train_, y_train)
            
            print('....... Building and evaluating models .......')
            if sensitivity_metric=='ic50':
                nn_hb = neural_net_with_hyperband(
                    X_train_, y_train_, X_val_, y_val_, X_test, y_test, data_dir,
                    step_size_nodes, min_nodes, max_nodes, max_trials, executions_per_trial,
                    patience, min_delta, epochs, learning_rate_min, learning_rate_max, output_dir_feature,
                    feature_select, metadata, plt_confusion=confusion, plt_rocauc=rocauc)
                rf = random_forest(
                    X_train_, y_train_, X_test, y_test, output_dir_feature, feature_select, metadata,
                    plt_confusion=confusion, plt_rocauc=rocauc)
                ridge = ridge_classifier(
                    X_train_, y_train_, X_test, y_test, output_dir_feature, feature_select, metadata,
                    plt_confusion=confusion, plt_rocauc=rocauc)
                evaluation_df = pd.concat([nn_hb, rf, ridge])
                evaluation_df.columns = ['model', 'tissue', 'acc', 'rocauc', 'n_correctly_predicted_sensitive_cell_lines', \
                    'n_correctly_predicted_resistant_cell_lines']
                evaluation_df.to_csv(output_dir_feature + '/evaluation_df.csv', index=False)
            
            print('....... Generating evaluation reports ........')
            plot_combined_rocauc(evaluation_df, feature_select, output_dir_feature)
            plot_combined_acc(evaluation_df, feature_select, output_dir_feature)
        stitch_pngs(output_dir, data_type)
            
    # else:
    #     # nn_hb = neural_net_with_hyperband_regression(
    #     #     X_train_ros, y_train_ros, X_val_, y_val_, X_test, y_test, data_dir,
    #     #     step_size_nodes, min_nodes, max_nodes, max_trials, executions_per_trial,
    #     #     patience, min_delta, epochs, learning_rate_min, learning_rate_max, output_dir,
    #     #     feature_selection, metadata, plt_confusion=confusion, plt_rocauc=rocauc)
    #     # rf = random_forest_regression(
    #     #     X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection, metadata,
    #     #     plt_confusion=confusion, plt_rocauc=rocauc)
    #     ridge = ridge_regression(
    #         X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection, metadata,
    #         plt_confusion=confusion, plt_rocauc=rocauc)
    #     evaluation_df = ridge
    #     # elastic = elastic_net(
    #     #     X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection, metadata,
    #     #     plt_confusion=confusion, plt_rocauc=rocauc)
    #     # evaluation_df = pd.concat([nn_hb, rf, ridge, elastic])
    
    # print(evaluation_df)

if __name__=='__main__': 
    main()
