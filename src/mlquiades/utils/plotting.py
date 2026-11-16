import os
import markdown
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay

LEFT_IN = 0.9     # space for the y-axis label
RIGHT_IN = 1.6    # space for the legend
GROUP_IN = 1.5    # width of one tissue group
TOP_IN = 0.4
BOTTOM_IN = 2.2   # space for model + tissue labels
DPI = 100

MODEL_NAMES = {'nn_hb': 'nn', 'ridge_classification': 'ridge',
               'svm': 'svm2', 'svm_l1': 'svm1'}


def tissue_order(tissues):
    '''
    Fixed tissue order used by every plot: all_tissues first, then alphabetical.
    Pass the same list (e.g. tissue_order(metadata['tissue'])) to every plot so
    they all have the same slots, even if a tissue has no test samples.
    '''
    rest = sorted(set(tissues) - {'all_tissues'})
    return ['all_tissues'] + rest


def _tissue_figure(n_tissues, n_rows=1, row_in=3.5, hspace_in=0.6):
    '''
    Figure whose axes span exactly n_tissues slots with fixed margins.
    Returns fig and a list of axes (one per row, sharing x).
    '''
    width = LEFT_IN + n_tissues * GROUP_IN + RIGHT_IN
    height = TOP_IN + n_rows * row_in + (n_rows - 1) * hspace_in + BOTTOM_IN
    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height), sharex=True,
                             squeeze=False)
    fig.subplots_adjust(left=LEFT_IN / width, right=1 - RIGHT_IN / width,
                        bottom=BOTTOM_IN / height, top=1 - TOP_IN / height,
                        hspace=hspace_in / row_in)
    axes = list(axes[:, 0])
    for ax in axes:
        ax.set_xlim(-0.5, n_tissues - 0.5)
        # dotted separators between tissue groups
        for i in range(1, n_tissues):
            ax.axvline(i - 0.5, color='black', linestyle='dotted', linewidth=1)
    return fig, axes


def _bar_offsets(n_bars, total=0.8):
    '''x offsets (relative to the group center) and width for n bars.'''
    w = total / n_bars
    return -total / 2 + w * (np.arange(n_bars) + 0.5), w


def _tissue_ticklabels(ax, tissues):
    '''Tissue names under the group centers (used when there are no model labels).'''
    ax.set_xticks(np.arange(len(tissues)))
    ax.set_xticklabels(tissues, rotation=45, ha='right', rotation_mode='anchor',
                       fontsize=9)


def _tissue_textlabels(ax, tissues, y=-0.18):
    '''Tissue names on a second row, below the model names.'''
    for i, tissue in enumerate(tissues):
        ax.text(i, y, tissue, transform=ax.get_xaxis_transform(), rotation=45,
                ha='right', va='top', rotation_mode='anchor', fontsize=9)


def plot_split(df, output_dir, tissues=None):
    '''
    Plot the split for training, validation and testing data with respect to the
    number of sensitive and resistant cancer cell lines in each tissue type.
    '''
    if tissues is None:
        tissues = tissue_order(df['tissue'])
    variables = list(df['variable'].unique())
    colors = sns.color_palette('icefire')
    offsets, w = _bar_offsets(len(variables), total=0.7)

    fig, axes = _tissue_figure(len(tissues), n_rows=3, row_in=3)
    ymax = df['value'].max()

    for ax, split, title in zip(axes, ['train', 'val', 'test'], ['TRAIN', 'VAL', 'TEST']):
        d = df[df['train_val_test'] == split]
        for k, var in enumerate(variables):
            vals = (d[d['variable'] == var].set_index('tissue')['value']
                    .reindex(tissues).to_numpy(dtype=float))
            x = np.arange(len(tissues)) + offsets[k]
            p = ax.bar(x, np.nan_to_num(vals), w, color=colors[k], label=var)
            ax.bar_label(p, labels=['' if np.isnan(v) else str(int(v)) for v in vals],
                         fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_ylabel('# of cell lines')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    _tissue_ticklabels(axes[-1], tissues)
    axes[-1].set_xlabel('')

    fig.savefig(output_dir + '/data_split.png', dpi=DPI)
    plt.close(fig)

    return


def plot_combined_rocauc(
        evaluation_df, feature_selection, output_dir, tissues=None):
    '''
    Creates barplot of ROC AUC scores across all models. Tissues where ROC AUC
    is undefined (only one class in the testing dataset) are left blank. Saves
    as png.
    '''
    evaluation_df = evaluation_df.copy()
    evaluation_df.columns = ['model', 'tissue', 'acc', 'rocauc', 'n_0', 'n_1']
    evaluation_df['model'] = evaluation_df['model'].replace(MODEL_NAMES)
    evaluation_df['rocauc'] = pd.to_numeric(evaluation_df['rocauc'], errors='coerce')
    if tissues is None:
        tissues = tissue_order(evaluation_df['tissue'])
    model_names = sorted(evaluation_df['model'].unique())
    offsets, w = _bar_offsets(len(model_names))
    colors = sns.color_palette()

    fig, (ax,) = _tissue_figure(len(tissues))
    for k, model in enumerate(model_names):
        d = evaluation_df[evaluation_df['model'] == model]
        vals = d.set_index('tissue')['rocauc'].reindex(tissues).to_numpy(dtype=float)
        ax.bar(np.arange(len(tissues)) + offsets[k], vals, w, color=colors[k],
               label=model)   # NaN bars are simply not drawn
    ax.axhline(0.5, color='black', linestyle='dotted', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('ROC AUC')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    _tissue_ticklabels(ax, tissues)

    fig.savefig(output_dir + '/plt_rocauc_all_' + feature_selection + '.png', dpi=DPI)
    plt.close(fig)

    df__ = (evaluation_df.dropna(subset=['rocauc'])
            .drop(columns=['acc', 'n_0', 'n_1'])
            .melt(id_vars=['model', 'tissue']))
    df__ = df__[['model', 'variable', 'value', 'tissue']]
    df__.to_csv(output_dir + '/evaluation_df_rocauc.csv', index=False)


def plot_combined_acc(
        evaluation_df, feature_selection, output_dir, tissues=None):
    '''
    Creates barplot of acc scores across all models. Saves as png.
    '''
    evaluation_df = evaluation_df.copy()
    evaluation_df.columns = ['model', 'tissue', 'acc', 'rocauc', 'n_0', 'n_1']
    evaluation_df['model'] = evaluation_df['model'].replace(MODEL_NAMES)
    if tissues is None:
        tissues = tissue_order(evaluation_df['tissue'])
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

    model_names = sorted(df__['model'].unique())
    offsets, w = _bar_offsets(len(model_names))

    fig, (ax,) = _tissue_figure(len(tissues))

    x_all, labels_all = [], []
    for k, model in enumerate(model_names):
        d = df__[df__['model'] == model].set_index('tissue').reindex(tissues)
        x = np.arange(len(tissues)) + offsets[k]
        bottom = np.zeros(len(tissues))
        for count, (label, col) in enumerate([('sensitive', 'acc_class0'),
                                              ('resistant', 'acc_class1')]):
            vals = np.nan_to_num(d[col].to_numpy(dtype=float))
            n = d['n_' + str(count)].to_numpy(dtype=float)
            p = ax.bar(x, vals, w * 0.9, bottom=bottom, color=f'C{count}',
                       label=label if k == 0 else None)
            ax.bar_label(p, labels=[str(int(v)) if v > 0 else '' for v in n],
                         label_type='center', fontsize=7)
            bottom += vals
        x_all.extend(x)
        labels_all.extend([model] * len(tissues))

    # row 1: model names under each bar; row 2: tissue names under each group
    order = np.argsort(x_all)
    ax.set_xticks(np.array(x_all)[order])
    ax.set_xticklabels(np.array(labels_all)[order], rotation=90, fontsize=8)
    ax.tick_params(axis='x', length=0)
    _tissue_textlabels(ax, tissues)

    ax.set_ylim(0, 1)
    ax.set_ylabel('accuracy')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    fig.savefig(output_dir + '/plt_accuracy_all_' + feature_selection + '.png', dpi=DPI)
    plt.close(fig)
    df__.to_csv(output_dir + '/evaluation_df_acc.csv', index=False)


def plot_confusion_matrix(
        y_test, y_pred, output_dir, model_name, nn=False):
    '''
    Plots confusion matrix and saves as png.
    '''
    if not nn:
        ConfusionMatrixDisplay.from_predictions(
            np.array([y_test.astype(np.float32)]).T,
            (y_pred).astype(int), display_labels=[0,1])
        plt.savefig(output_dir + '/confusion/plt_confusion_' + model_name + '.png')
        plt.close()
    else:
        ConfusionMatrixDisplay.from_predictions(
            np.array([y_test.astype(np.float32)]).T,
            (y_pred>=.5).astype(int), display_labels=[0,1])
        plt.savefig(output_dir + '/confusion/plt_confusion_' + model_name + '.png')
        plt.close()

def stitch_pngs(
        output_dir, data_type):
    '''
    Puts resulting pngs from all feature selection methods into md and html reports.
    Images are shown at their natural size (no scaling) so tissue columns line up.
    Feature selections whose plots were not produced are skipped.
    '''
    report_str = ['# Report', '', '## Data Distribution', '']
    report_str.append('![data split](data_split.png)')

    feature_selections = ['cdk4_6_genes', 'cdk4_6_cancer', 'pearson']
    for feature_select in feature_selections:
        png = feature_select + '_' + data_type + '/plt_accuracy_all_' + feature_select + '.png'
        if not os.path.isfile(os.path.join(output_dir, png)):
            continue
        report_str.extend(['', '## Evaluation - Accuracy (' + feature_select \
                        + ')', ''])
        report_str.append('![acc all](' + png + ')')
    for feature_select in feature_selections:
        png = feature_select + '_' + data_type + '/plt_rocauc_all_' + feature_select + '.png'
        if not os.path.isfile(os.path.join(output_dir, png)):
            continue
        report_str.extend(['', '## Evaluation - ROCAUC (' + feature_select \
                        + ')', '### (for tissues that contain both classes in the testing dataset)', ''])
        report_str.append('![rocauc all](' + png + ')')

    with open(output_dir + '/report_' + data_type + '.md', 'w') as f:
        for item in report_str:
            f.write(item)
            f.write('\n')
    with open(output_dir + '/report_' + data_type + '.md', 'r') as f:
        md = f.read()
    html = markdown.markdown(md)
    # keep images at natural size so every plot uses the same pixel scale
    html = ('<html><head><style>img { max-width: none; display: block; }'
            '</style></head><body>' + html + '</body></html>')
    with open(output_dir + '/report_' + data_type + '.html', 'w') as f:
        f.write(html)
