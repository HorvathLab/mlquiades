import markdown
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_combined_rocauc(
        evaluation_df, feature_selection, output_dir):
    '''
    Creates barplot of ROC AUC scores across all models. Saves as png.
    '''
    evaluation_df = pd.DataFrame(evaluation_df)
    evaluation_df.columns = ['model', 'tissue', 'acc', 'f1', 'rocauc']

    df__ = evaluation_df.drop(columns=['acc', 'f1']).melt(id_vars=['model','tissue'])
    df__['value'] = np.array(df__['value'],dtype='float')
    df__ = pd.DataFrame({'model': df__['model'], 'variable': df__['variable'],
                         'value': df__['value'], 'tissue': df__['tissue']})

    bp = sns.barplot(df__, x='tissue', y='value', hue='model')
    bp.set(xlabel=' ', ylabel=' ', title=feature_selection)
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45)
    fig = bp.get_figure()
    plt.tight_layout()
    fig.savefig(output_dir + '/' + 'plt_rocauc_all_' + feature_selection + '.png')

def plot_confusion_matrix(
        y_test, y_pred, output_dir, feature_selection, model_name, nn=False):
    '''
    Plots confusion matrix and saves as png.
    '''
    if not nn:
        ConfusionMatrixDisplay.from_predictions(
            np.array([y_test.astype(np.float32)]).T,
            (y_pred).astype(int),display_labels=[0,1])
        plt.savefig(output_dir + '/' + 'plt_confusion_' + model_name + '_'
                    + feature_selection + '.png')
        plt.close()
    else:
        ConfusionMatrixDisplay.from_predictions(
            np.array([y_test.replace(-1,0).astype(np.float32)]).T,
            (y_pred>=.5).astype(int),display_labels=[0,1])
        plt.savefig(output_dir + '/' + 'plt_confusion_' + model_name + '_'
                    + feature_selection + '.png')
        plt.close()

def plot_rocauc(
        fpr, tpr, output_dir, feature_selection, model_name):
    '''
    Plots ROC AUC and saves as png.
    '''
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title('ROC AUC '+model_name+' '+feature_selection)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(output_dir + '/' + 'plt_rocauc_' + model_name + '_'
                + feature_selection +'.png')
    plt.close()

def stitch_pngs(
        feature_selection, output_dir):
    '''
    Puts all generated pngs into a single md report.
    '''
    model_list = ['dt','gbdt','nn','nn_hb','rf','ridge']
    model_list_full = ['Decision Tree', 'Gradient-Boosted Decision Tree',
            'Neural Net', 'Neural Net with Hyperband Tuning', 'Random Forest',
            'Ridge Classifier']
    report_str = ['# Final Report', '', 'legend: 0 = sensitive, 1 = resistant', '']
    for i in range(0,6):
        model_name = model_list[i]
        report_str.append('')
        report_str.append('## ' + model_list_full[i])
        report_str.append('![' + model_name + '_' + 'rocauc]' + '(plt_rocauc_' 
                + model_name + '_' + feature_selection + '.png)')
        report_str.append('![' + model_name + '_' + 'confusion]' + '(plt_confusion_'
                + model_name + '_' + feature_selection + '.png)')

    report_str.append('![rocauc all](plt_rocauc_all_' + feature_selection + '.png)')
    
    with open(output_dir + '/' + 'report.md', 'w') as f:
        for item in report_str:
            f.write(item)
            f.write('\n')
    with open(output_dir + '/' + 'report.md', 'r') as f:
        md = f.read()
    html = markdown.markdown(md)
    with open(output_dir + '/' + 'report.html', 'w') as f:
        f.write(html)



