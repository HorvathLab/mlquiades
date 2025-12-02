import markdown
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay

def plot_combined_rocauc(
        evaluation_df, feature_selection, output_dir):
    '''
    Creates barplot of ROC AUC scores across all models. Saves as png.
    '''
    evaluation_df.columns = ['model', 'tissue', 'acc', 'rocauc', 'n_0', 'n_1']
    evaluation_df = evaluation_df.drop(columns=['n_0','n_1'])
    evaluation_df = pd.DataFrame(evaluation_df.sort_values(by=['tissue', 'model'],
                                                           ascending=[True, True]))
    evaluation_df_all = evaluation_df[evaluation_df['tissue']=='all_tissues']
    evaluation_df_rest = evaluation_df[evaluation_df['tissue']!='all_tissues']
    evaluation_df = pd.concat([evaluation_df_all, evaluation_df_rest])
    
    df__ = evaluation_df.drop(columns=['acc']).melt(id_vars=['model','tissue'])
    df__['value'] = np.array(df__['value'],dtype='float')
    df__ = pd.DataFrame({'model': df__['model'], 'variable': df__['variable'],
                         'value': df__['value'], 'tissue': df__['tissue']})

    bp = sns.barplot(df__, x='tissue', y='value', hue='model')
    handles, _ = bp.get_legend_handles_labels()
    plt.legend(handles[0:6], ['dt', 'gbdt', 'nn', 'nn_hb', 'rf', 'ridge'], loc="lower left", 
               bbox_to_anchor=(1.01, 0.29), title="Model")
    bp.set(xlabel=' ', ylabel=' ', title='ROCAUC Evaluation Using ' + feature_selection + 
           ' Features')
    bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
    fig = bp.get_figure()
    plt.tight_layout()
    fig.savefig(output_dir + '/all_tissues/plt_rocauc_all_' + feature_selection + '.png')

def plot_combined_acc(
        evaluation_df, feature_selection, output_dir):
    '''
    Creates barplot of acc scores across all models. Saves as png.
    '''
    evaluation_df.columns = ['model', 'tissue', 'acc', 'rocauc', 'n_0', 'n_1']
    evaluation_df = pd.DataFrame(evaluation_df.sort_values(by=['tissue', 'model'],
                                                           ascending=[True, True]))
    evaluation_df_all = evaluation_df[evaluation_df['tissue']=='all_tissues']
    evaluation_df_rest = evaluation_df[evaluation_df['tissue']!='all_tissues']
    evaluation_df = pd.concat([evaluation_df_all, evaluation_df_rest])
    evaluation_df.to_csv(output_dir + '/all_tissues/evaluation.csv', index=False)    
    df__ = evaluation_df
    df__['n_0_plus_n_1'] = (df__['n_0'] + df__['n_1'])

    df__.loc[df__['n_0_plus_n_1']==0, 'n_0'] = -1
    df__.loc[df__['n_0_plus_n_1']==0, 'n_1'] = -1

    df__['acc_class0'] = df__['acc'] * (df__['n_0']/(df__['n_0']+df__['n_1']))
    df__['acc_class1'] = df__['acc'] * (df__['n_1']/(df__['n_0']+df__['n_1']))

    df__.loc[df__['n_0_plus_n_1']==0, 'acc_class0'] = 0
    df__.loc[df__['n_0_plus_n_1']==0, 'acc_class1'] = 0

    df__.loc[df__['n_0_plus_n_1']==0, 'n_0'] = 0
    df__.loc[df__['n_0_plus_n_1']==0, 'n_1'] = 0

    df__ = df__.drop(columns=['acc', 'n_0_plus_n_1', 'rocauc'])
    
    tissues = df__['tissue'].unique()
    
    fig, ax = plt.subplots(1,1, sharex='col', sharey='row',figsize=(1.5*16, 4), tight_layout=True)
    models = (
        'nn',
        'rf\n' + tissues[0],
        'ridge',
        'nn\n',
        'rf\n ' + tissues[1],
        'ridge\n',
        'nn\n ',
        'rf\n' + tissues[2],
        'ridge\n  ',
        'nn\n  ',
        'rf\n' + tissues[3],
        'ridge\n   ',
        'nn\n    ',
        'rf\n' + tissues[4],
        'ridge\n    ',
        'nn\n     ',
        'rf\n' + tissues[5],
        'ridge\n     ',
        'nn\n      ',
        'rf\n' + tissues[6],
        'ridge\n      ',
        'nn\n       ',
        'rf\n' + tissues[7],
        'ridge\n       ',
        'nn\n        ',
        'rf\n' + tissues[8],
        'ridge\n        ',
        'nn\n         ',
        'rf\n' + tissues[9],
        'ridge\n         ',
        'nn\n          ',
        'rf\n' + tissues[10],
        'ridge\n          ',
        'nn\n           ',
        'rf\n' + tissues[11],
        'ridge\n           ',
        'nn\n            ',
        'rf\n' + tissues[12],
        'ridge\n            ',
        'nn\n             ',
        'rf\n' + tissues[13],
        'ridge\n             ',
        'nn\n              ',
        'rf\n' + tissues[14],
        'ridge\n              ',
        'nn\n               ',
        'rf\n' + tissues[15],
        'ridge\n               ',
        'nn\n                ',
        'rf\n' + tissues[16],
        'ridge\n                ',
        'nn\n                 ',
        'rf\n' + tissues[17],
        'ridge\n                 '
    )
    weight_counts = {
        'sensitive': df__[['acc_class0']].to_numpy().flatten(),
        'resistant': df__[['acc_class1']].to_numpy().flatten(),
    }
    count = 0
    width = 0.5
    bottom = np.zeros(54)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(models, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
        ax.bar_label(p, labels=df__['n_'+str(count)].to_numpy().flatten(), label_type='center')
        count += 1

    ax.legend(loc="upper right")
    plt.ylabel('accuracy')
    for i in np.arange(17):
        plt.vlines(2.5+(3*i), 0, 1, colors='black', linestyles='dotted')

    plt.savefig(output_dir + '/all_tissues/plt_accuracy_all_' + feature_selection + '.png')

def plot_confusion_matrix(
        y_test, y_pred, output_dir, feature_selection, model_name, nn=False):
    '''
    Plots confusion matrix and saves as png.
    '''
    if not nn:
        ConfusionMatrixDisplay.from_predictions(
            np.array([y_test.astype(np.float32)]).T,
            (y_pred).astype(int),display_labels=[0,1])
        plt.savefig(output_dir + '/by_tissue/confusion/plt_confusion_' + model_name + '_'
                    + feature_selection + '.png')
        plt.close()
    else:
        ConfusionMatrixDisplay.from_predictions(
            np.array([y_test.replace(-1,0).astype(np.float32)]).T,
            (y_pred>=.5).astype(int),display_labels=[0,1])
        plt.savefig(output_dir + '/by_tissue/confusion/plt_confusion_' + model_name + '_'
                    + feature_selection + '.png')
        plt.close()

def plot_rocauc(
        fpr, tpr, output_dir, feature_selection, model_name):
    '''
    Plots ROC AUC and saves as png.
    '''
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title('ROC AUC ' + model_name + ' ' + feature_selection)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(output_dir + '/by_tissue/rocauc/plt_rocauc_' + model_name + '_'
                + feature_selection + '.png')
    plt.close()

def stitch_pngs(
        feature_selection, output_dir):
    '''
    Puts all generated pngs into a single md report.
    '''
    model_list = ['nn_hb','rf','ridge'] #'dt','gbdt', 'nn',
    model_list_full = ['Neural Net with Hyperband Tuning', 'Random Forest',
            'Ridge Classifier'] # 'Decision Tree', 'Gradient-Boosted Decision Tree', 'Neural Net',
    report_str = ['# Final Report'] #, '', 'legend: 0 = sensitive, 1 = resistant', '']
    # for i in range(0, 3):
    #     model_name = model_list[i]
    #     report_str.append('')
    #     report_str.append('## ' + model_list_full[i])
    #     report_str.append('![' + model_name + '_' + 'rocauc]' + '(' + output_dir
    #                       + / plt_rocauc_' 
    #                       + model_name + '_' + feature_selection + '.png)')
    #     report_str.append('![' + model_name + '_' + 'confusion]' + '(' + output_dir
    #                       + '/all_tissues/confusion/plt_confusion_' + model_name + '_'
    #                       + feature_selection + '.png)')

    report_str.append('![acc all](' + output_dir + 'all_tissues/plt_accuracy_all_'
                      + feature_selection + '.png)')
    report_str.append('![acc all](' + output_dir + 'all_tissues/plt_rocauc_all_'
                      + feature_selection + '.png)')
        
    with open(output_dir + '/all_tissues/report.md', 'w') as f:
        for item in report_str:
            f.write(item)
            f.write('\n')
    with open(output_dir + '/all_tissues/report.md', 'r') as f:
        md = f.read()
    html = markdown.markdown(md)
    with open(output_dir + '/all_tissues/report.html', 'w') as f:
        f.write(html)
