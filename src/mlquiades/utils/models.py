import keras
import keras_tuner
import numpy as np
import pandas as pd
from .plotting import *
from keras import layers
from sklearn import metrics
from itertools import compress
from tensorflow.keras.layers import Dense
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def evaluate(
        y_test, y_pred):
    '''
    Evaluates y-label predictions for testing dataset. Returns acc, false
    positive rate, true positive rate and ROCAUC values. This is only for non-keras
    models.
    '''
    acc = metrics.accuracy_score(y_test, y_pred)
    # f1 = f1_score(np.array(y_test), (y_pred).astype(int))    
    fpr, tpr, thresholds = roc_curve(y_test,y_pred)
    rocauc = roc_auc_score(y_test,y_pred)
    
    return acc, rocauc.item(), fpr, tpr

def evaluate_keras(
        model, X_test, y_test, y_pred):
    '''
    Evaluates y-label predictions for testing dataset. Returns acc, false
    positive rate, true positive rate and ROCAUC values. This is only for keras
    models.
    '''
    loss, acc = model.evaluate(X_test, y_test)
    fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
    rocauc = auc(fpr, tpr)
    # f1_metric = keras.metrics.F1Score(threshold=0.5)
    # f1_metric.update_state(np.array([y_test.astype(np.float32)]).T,y_pred)
    
    return acc, rocauc, fpr, tpr

def counter(y_test, y_pred, nn=False):
    '''
    Count the number of correct 1s and 0s.
    '''
    y_test = y_test.to_numpy()
    y_pred = y_pred.flatten()
    if nn:
        y_pred = (y_pred>=.5).astype(int)
    counter = list(compress(y_test, np.equal(y_test,y_pred).tolist()))
    zeros = counter.count(0) + counter.count(-1) # count the number of correctly predicted sensitive samples
    ones = counter.count(1) # count the number of correctly predicted resistant samples
    
    return zeros, ones

def neural_net_with_hyperband(
        X_train_ros, y_train_ros, X_val_, y_val_, X_test, y_test, data_dir,
        step_size_nodes, min_nodes, max_nodes, max_trials, executions_per_trial,
        patience, min_delta, epochs, learning_rate_min, learning_rate_max, output_dir,
        feature_selection, metadata, plt_confusion=False, plt_rocauc=False):
    '''
    Builds hyperband-tuned neural net using keras. Fits the model to the randomly
    oversampled training data. Makes predictions on the testing dataset. Plots
    confusion matrix and ROCAUC plot.
    '''
    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('n_layers', min_value=1, max_value=20)):
            model.add(Dense(units=hp.Int('units__' + str(i), min_value=min_nodes,
                                         max_value=max_nodes, step=step_size_nodes), activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        learning_rate = hp.Float('learning_rt', min_value=learning_rate_min,
                                 max_value=learning_rate_max, sampling='log')
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    y_test = y_test.replace(-1,0)
    build_model(keras_tuner.HyperParameters())

    tuner = keras_tuner.RandomSearch(hypermodel=build_model,
                                     objective='val_accuracy', max_trials=max_trials, 
                                     executions_per_trial=executions_per_trial, overwrite=True,
                                     directory=data_dir)
    #tuner.search_space_summary()
    
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience, min_delta=min_delta)
    tuner.search(X_train_ros, y_train_ros.replace(-1,0), epochs=epochs,
                validation_data=(X_val_, y_val_.replace(-1,0)), callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue]
        # y_test_tissue = y_test_tissue.replace(-1,0)
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = best_model.predict(X_test_tissue)
            if len(y_test_tissue['label'].unique())>1:          
                acc, rocauc, fpr, tpr = evaluate_keras(best_model, X_test_tissue,
                                                       y_test_tissue['label'], y_pred,
                                                       )
                zeros, ones = counter(y_test_tissue['label'], y_pred, nn=True)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                #                         model_name='nn_hb_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn_hb_'
                #                 + tissue)
                evaluation_df.append(['nn_hb', tissue, acc, rocauc, zeros, ones])
            else:
                loss, acc = best_model.evaluate(X_test_tissue, y_test_tissue)
                zeros, ones = counter(y_test_tissue['label'], y_pred, nn=True)
                evaluation_df.append(['nn_hb', tissue, acc, 0, zeros, ones])
    
    y_pred = best_model.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate_keras(best_model, X_test, y_test['label'], y_pred, 
                                           )
    zeros, ones = counter(y_test['label'], y_pred, nn=True)
    evaluation_df.append(['nn_hb', 'all_tissues', acc, rocauc, zeros, ones])
    
    # if plt_confusion:
    #     plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
    #                         model_name='nn_hb', nn=True)
    # if plt_rocauc:
    #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn_hb')
    
    return pd.DataFrame(evaluation_df)


def neural_net_with_hyperband_regression(
        X_train, y_train, X_val_, y_val_, X_test, y_test, data_dir,
        step_size_nodes, min_nodes, max_nodes, max_trials, executions_per_trial,
        patience, min_delta, epochs, learning_rate_min, learning_rate_max, output_dir,
        feature_selection, metadata, plt_confusion=False, plt_rocauc=False):
    '''
    Builds hyperband-tuned neural net classifiear using keras. Fits the model to 
    training data. Makes predictions on the testing dataset. Plots confusion matrix
    and ROCAUC plot.
    '''
    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('n_layers', min_value=1, max_value=20)):
            model.add(Dense(units=hp.Int('units__' + str(i), min_value=min_nodes,
                                         max_value=max_nodes, step=step_size_nodes), activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        learning_rate = hp.Float('learning_rt', min_value=learning_rate_min,
                                 max_value=learning_rate_max, sampling='log')
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mse'])
        return model
    
    y_test = y_test
    build_model(keras_tuner.HyperParameters())

    tuner = keras_tuner.RandomSearch(hypermodel=build_model,
                                     objective='val_accuracy', max_trials=max_trials, 
                                     executions_per_trial=executions_per_trial, overwrite=True,
                                     directory=data_dir)
    
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience, min_delta=min_delta)
    tuner.search(X_train, y_train, epochs=epochs,
                validation_data=(X_val_, y_val_), callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = best_model.predict(X_test_tissue)
            if len(y_test_tissue['label'].unique())>1:          
                acc, rocauc, fpr, tpr, rmse = evaluate_keras(best_model, X_test_tissue,
                                                       y_test_tissue['label'], y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred, nn=True)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                #                         model_name='nn_hb_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn_hb_'
                #                 + tissue)
                evaluation_df.append(['nn_hb', tissue, acc, rocauc, zeros, ones])
            else:
                loss, acc = best_model.evaluate(X_test_tissue, y_test_tissue)
                zeros, ones = counter(y_test_tissue['label'], y_pred, nn=True)
                evaluation_df.append(['nn_hb', tissue, acc, 0, zeros, ones])
    
    y_pred = best_model.predict(X_test)
    
    acc, rocauc, fpr, tpr, rmse = evaluate_keras(best_model, X_test, y_test['label'], y_pred)
    zeros, ones = counter(y_test['label'], y_pred, nn=True)
    evaluation_df.append(['nn_hb', 'all_tissues', acc, rocauc, zeros, ones])
    
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                            model_name='nn_hb', nn=True)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn_hb')
    
    return pd.DataFrame(evaluation_df)

def random_forest(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection,
        metadata, max_depth=2, random_state=0, plt_confusion=False, plt_rocauc=False):
    '''
    Builds random forest model. Fits the model to the randomly oversampled
    training data. Makes predictions on the testing dataset. Plots confusion
    matrix and ROCAUC plot.
    '''
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf = clf.fit(X_train_ros, y_train_ros)
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)         
            if len(y_test_tissue['label'].unique())>1:            
                acc, rocauc, fpr, tpr = evaluate(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir,
                #                           feature_selection,
                #                         model_name='rf_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='rf_'
                #                 + tissue)
                evaluation_df.append(['rf', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['rf', tissue, acc, 0, zeros, ones])

    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred)
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['rf', 'all_tissues', acc, rocauc, zeros, ones])
    
    # if plt_confusion:
    #     plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
    #                           model_name='rf', nn=False)
    # if plt_rocauc:
    #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='rf')
    
    return pd.DataFrame(evaluation_df)


def random_forest_regression(
        X_train, y_train, X_test, y_test, output_dir, feature_selection,
        metadata, max_depth=2, random_state=0, plt_confusion=False, plt_rocauc=False):
    '''
    Builds random forest regression model. Fits the model to training data. Makes 
    predictions on the testing dataset. Plots confusion matrix and ROCAUC plot.
    '''
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf = clf.fit(X_train, y_train)
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)         
            if len(y_test_tissue['label'].unique())>1:            
                acc, rocauc, fpr, tpr, rmse = evaluate(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir,
                #                           feature_selection,
                #                         model_name='rf_regression_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='rf_'
                #                 + tissue)
                evaluation_df.append(['rf', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['rf_regression', tissue, acc, 0, zeros, ones])

    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr, rmse = evaluate(y_test, y_pred)
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['rf_regression', 'all_tissues', acc, rocauc, zeros, ones])
    
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                              model_name='rf_regression', nn=False)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='rf')
    
    return pd.DataFrame(evaluation_df)


def ridge_classifier(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection,
        metadata, plt_confusion=False, plt_rocauc=False):
    '''
    Builds ridge classifier model. Fits the model to the randomly oversampled
    training data. Makes predictions on the testing dataset. Plots confusion
    matrix and ROCAUC plot.
    '''
    clf = RidgeClassifier().fit(X_train_ros, y_train_ros)
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)
            if len(y_test_tissue['label'].unique())>1:
                acc, rocauc, fpr, tpr = evaluate(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, 
                #                           feature_selection,
                #                         model_name='ridge_classification_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='ridge_'
                #                 + tissue)
                evaluation_df.append(['ridge_classification', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['ridge_classification', tissue, acc, 0, zeros, ones])
    
    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred)
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['ridge_classification', 'all_tissues', acc, rocauc, zeros, ones])
    
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                              model_name='ridge_classification', nn=False)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='ridge_classification')
    
    return pd.DataFrame(evaluation_df)

def ridge_regression(
    X_train, y_train, X_test, y_test, output_dir, feature_selection,
        metadata, plt_confusion=False, plt_rocauc=False):
    '''
    Builds ridge regression model (linear least squares with l2 regularization).
    '''
    
    clf = Ridge().fit(X_train, y_train)
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)
            if len(y_test_tissue['label'].unique())>1:
                acc, rocauc, fpr, tpr, rmse = evaluate(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, 
                #                           feature_selection,
                #                         model_name='ridge_regression_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='ridge_'
                #                 + tissue)
                evaluation_df.append(['ridge_regression', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['ridge_regression', tissue, acc, 0, zeros, ones])
    
    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr, rmse = evaluate(y_test, y_pred)
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['ridge_regression', 'all_tissues', acc, rocauc, zeros, ones])

def elastic_net(
    X_train, y_train, X_test, y_test, output_dir, feature_selection,
    metadata, plt_confusion=False, plt_rocauc=False):
    '''
    Builds elastic net model (l1 and l2 regularization).
    '''
    
    clf = ElasticNet().fit(X_train, y_train)
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)
            if len(y_test_tissue['label'].unique())>1:
                acc, rocauc, fpr, tpr, rmse = evaluate(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, 
                #                           feature_selection,
                #                         model_name='elastic_net_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='ridge_'
                #                 + tissue)
                evaluation_df.append(['elastic_net', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['elastic_net', tissue, acc, 0, zeros, ones])
    
    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr, rmse = evaluate(y_test, y_pred)
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['elastic_net', 'all_tissues', acc, rocauc, zeros, ones])
