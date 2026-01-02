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
        dest='feature_selection',
        default=None,
        help='choose one of three options for feature selection for the ml models,' \
        'options include: cdk4_6_genes, cdk4_6_cancer_genes, pearson (required)')
    parser.add_argument(
        '--s',
        type=str,
        action='store',
        dest='cdk4_6_genes_filename',
        help='filename for cdk4 and cdk6 genes for the feature selection option cdk4_6_genes (optional if pearson selected for -r)')
    parser.add_argument(
        '--t',
        type=str,
        action='store',
        dest='cancer_genes_filename',
        help='filename for cancer genes for the feature selection option cdk4_6_cancer_genes (optional unless cdk4_6_cancer_genes selected for -r)')
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

    return parser

def main():
    parser = params()
    args = parser.parse_args()
    data_dir = args.data_dir + '/'
    output_dir = args.output_folder_name
    data_filename = args.data_filename
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
    feature_selection = args.feature_selection
    cdk4_6_filename = args.cdk4_6_genes_filename
    cancer_genes_filename = args.cancer_genes_filename
    drug = args.drug
    gdsc = args.gdsc
    drug = drug.lower()
    gdsc = gdsc.lower()
    
    if feature_selection is None:
        raise TypeError('missing feature selection (option --r)')
    if feature_selection == 'cdk4_6_genes':
        if cdk4_6_filename is None:
            raise TypeError('missing cdk4_6_genes_filename (option --s)')
    if feature_selection == 'cdk4_6_cancer_genes':
        if cancer_genes_filename is None:
            raise TypeError('missing cancer_genes_filename (option --t)')           

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir + '/all_tissues/'):
        os.mkdir(output_dir + '/all_tissues')
    if rocauc or confusion:
        if not os.path.isdir(output_dir + '/by_tissue/'):
            os.mkdir(output_dir + '/by_tissue')
    if confusion:
        if not os.path.isdir(output_dir + '/by_tissue/confusion'):
            os.mkdir(output_dir + '/by_tissue/confusion')
    if rocauc:
        if not os.path.isdir(output_dir + '/by_tissue/rocauc'):
            os.mkdir(output_dir + '/by_tissue/rocauc')
    
    print('....... Reading in data .......')
    df = pd.read_csv(data_dir + data_filename)
    df['label'] = df['label_' + drug + '_' + gdsc]
    columns_label = [x for x in df.columns if 'label_' in x]
    df = df.dropna(subset=['label_' + drug + '_' + gdsc]).drop(
        columns=columns_label)
    
    print('....... Splitting and scaling data .......')
    X_train_ros, y_train_ros, X_val_, y_val_, X_test, y_test, metadata = split_scale_data(
        data_dir=data_dir, output_dir=output_dir, df=df, ros=ros,
        feature_selection=feature_selection, cdk4_6_genes_filename=cdk4_6_filename,
        cancer_genes_filename=cancer_genes_filename)

    print('....... Building and evaluating models .......')
    nn_hb = neural_net_with_hyperband(
        X_train_ros, y_train_ros, X_val_, y_val_, X_test, y_test, data_dir,
        step_size_nodes, min_nodes, max_nodes, max_trials, executions_per_trial,
        patience, min_delta, epochs, learning_rate_min, learning_rate_max, output_dir,
        feature_selection, metadata, plt_confusion=confusion, plt_rocauc=rocauc)
    rf = random_forest(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection, metadata,
        plt_confusion=confusion, plt_rocauc=rocauc)
    ridge = ridge_classifier(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection, metadata,
        plt_confusion=confusion, plt_rocauc=rocauc)
    evaluation_df = pd.concat([nn_hb, rf, ridge])
    
    print('....... Generating evaluation reports .......')
    plot_combined_acc(evaluation_df, feature_selection, output_dir)
    stitch_pngs(feature_selection, output_dir)

if __name__=='__main__': 
    main()
