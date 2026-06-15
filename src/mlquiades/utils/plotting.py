import markdown
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay


def plot_split(df, feature_selection, output_dir):
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
    p2.set_ylim(top=df['value'].max()+20)
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
    plt.close()
    
    return

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
    df__['value'] = np.array(df__['value'], dtype='float')
    df__ = pd.DataFrame({'model': df__['model'], 'variable': df__['variable'],
                         'value': df__['value'], 'tissue': df__['tissue']})
    bp = sns.barplot(df__, x='tissue', y='value', hue='model')
    fig = bp.get_figure()
    fig.set_size_inches((1.5*16, 4))
    # plt.subplots(figsize=(20,20))
    plt.tight_layout()
    fig.savefig(output_dir + '/all_tissues/plt_rocauc_all_' + feature_selection + '.png')
    plt.close()

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

    fig, ax = plt.subplots(1,1, sharex='col', sharey='row', figsize=(1.5*16, 4), tight_layout=True)

    spacer = ''
    models = list()

    for i in np.arange(len(tissues)):
        models.extend(
            ['nn\n' + spacer,
            'rf\n' + tissues[i],
            'ridge\n' + spacer]
        )
        spacer = spacer + ' '

    weight_counts = {
        'sensitive': df__[['acc_class0']].to_numpy().flatten(),
        'resistant': df__[['acc_class1']].to_numpy().flatten(),
    }
    count = 0
    width = 0.5
    bottom = np.zeros(len(tissues)*3)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(models, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
        ax.bar_label(p, labels=df__['n_' + str(count)].to_numpy().flatten(),
                        label_type='center')
        count += 1

    ax.legend(loc="upper right")
    plt.ylabel('accuracy')
    for i in np.arange(len(tissues)):
        plt.vlines(2.5+(3*i), 0, 1, colors='black', linestyles='dotted')
    
    plt.savefig(output_dir + '/all_tissues/plt_accuracy_all_' + feature_selection + '.png')
    plt.close()

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
    Puts resulting pngs into an md and html reports.
    '''
    model_list = ['nn_hb','rf','ridge']
    model_list_full = ['Neural Net with Hyperband Tuning', 'Random Forest',
                       'Ridge Classifier']
    report_str = ['# Report', '', '## Data Distribution', '']
    report_str.append('![data split](all_tissues/data_split.png)')
    report_str.extend(['', '## Evaluation - Accuracy', ''])
    report_str.append('![acc all](all_tissues/plt_accuracy_all_' + feature_selection \
        + '.png)')
    report_str.extend(['', '## Evaluation - ROCAUC', '### (for tissues that contain both \
                       classes in the testing dataset)', ''])
    report_str.append('![rocauc all](all_tissues/plt_rocauc_all_' + feature_selection \
        + '.png)')
    
    with open(output_dir + '/report.md', 'w') as f:
        for item in report_str:
            f.write(item)
            f.write('\n')
    with open(output_dir + '/report.md', 'r') as f:
        md = f.read()
    html = markdown.markdown(md)
    with open(output_dir + '/report.html', 'w') as f:
        f.write(html)
