import math
import keras
import keras_tuner
import numpy as np
import pandas as pd
from .plotting import *
from sklearn import svm
from keras import layers
from sklearn import metrics
from sklearn.base import clone
from itertools import compress
from scipy.stats import loguniform, randint
from tensorflow.keras.layers import Dense
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.experimental import enable_halving_search_cv # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.linear_model import Ridge, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

MIN_PER_CLASS = 3

def has_enough_samples(labels, min_per_class=MIN_PER_CLASS):
    '''
    True if the tissue has both classes in the testing dataset, with at least
    min_per_class samples each; otherwise ROCAUC is undefined or too noisy.
    '''
    counts = labels.value_counts()
    return len(counts) > 1 and counts.min() >= min_per_class



def hyperband_search(
        estimator, param_distributions, X_train, y_train, X_val=None, y_val=None,
        resource='n_samples', max_resource=None, min_resource=None, eta=3,
        scoring='roc_auc', random_state=0, n_jobs=-1):
    '''
    Hyperband tuning for sklearn estimators. Runs one successive halving bracket
    (HalvingRandomSearchCV) per Hyperband bracket, trading off the number of
    sampled configurations against the starting budget, and keeps the best
    configuration across all brackets. The budget is either the number of
    training samples ('n_samples') or an estimator parameter such as
    'n_estimators'. If a validation set is given, candidates are trained on the
    training data and scored on the validation data; otherwise 5-fold stratified
    CV on the training data is used. Returns the best estimator refit on the
    full training data, its params and its validation score.
    '''
    X_train = np.asarray(X_train)
    y_train = np.ravel(np.asarray(y_train))
    n_classes = len(np.unique(y_train))

    if X_val is not None:
        X_val = np.asarray(X_val)
        y_val = np.ravel(np.asarray(y_val))
        X = np.concatenate((X_train, X_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=None)
        cv = [(np.arange(X_train.shape[0]), np.arange(X_train.shape[0], X.shape[0]))]
    else:
        X, y = X_train, y_train
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if resource == 'n_samples':
        # halving subsamples the training indices of each split by
        # n_resources / X.shape[0], so the budget is expressed in terms of X
        max_resource = X.shape[0] if max_resource is None else min(max_resource, X.shape[0])
        min_resource = 20 * n_classes if min_resource is None else min_resource
    else:
        max_resource = 500 if max_resource is None else max_resource
        min_resource = 10 if min_resource is None else min_resource
    min_resource = min(min_resource, max_resource)

    s_max = int(math.floor(math.log(max_resource / min_resource, eta) + 1e-9))
    best_score, best_params = -np.inf, None
    for s in range(s_max, -1, -1):
        n_candidates = int(math.ceil((s_max + 1) / (s + 1) * eta**s))
        r = max(int(max_resource * eta**(-s)), min_resource)
        search = HalvingRandomSearchCV(
            estimator, param_distributions, n_candidates=n_candidates, factor=eta,
            resource=resource, min_resources=r, max_resources=max_resource,
            cv=cv, scoring=scoring, refit=False, random_state=random_state + s,
            n_jobs=n_jobs)
        search.fit(X, y)
        if search.best_score_ > best_score:
            best_score, best_params = search.best_score_, search.best_params_

    best_model = clone(estimator)
    if best_params is not None:
        best_model.set_params(**best_params)
    if resource != 'n_samples':
        best_model.set_params(**{resource: max_resource})
    best_model.fit(X_train, y_train)

    return best_model, best_params, best_score

def predict_score(clf, X):
    '''
    Returns continuous scores for the positive class (probabilities if the
    model has them, otherwise decision function values) for ROCAUC.
    '''
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    return clf.decision_function(X)

def evaluate(
        y_test, y_pred, y_score):
    '''
    Evaluates y-label predictions for testing dataset. Accuracy is computed from
    the predicted labels, false positive rate, true positive rate and ROCAUC from
    the continuous scores. This is only for non-keras models.
    '''
    acc = metrics.accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    rocauc = roc_auc_score(y_test, y_score)

    return acc, rocauc.item(), fpr, tpr

def evaluate_keras(
        model, X_test, y_test, y_pred):
    '''
    Evaluates y-label predictions for testing dataset. Returns acc, false
    positive rate, true positive rate and ROCAUC values. This is only for keras
    models.
    '''
    acc = model.evaluate(X_test, y_test, return_dict=True)['accuracy']
    fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
    rocauc = auc(fpr, tpr)
    
    return acc, rocauc, fpr, tpr

def counter(y_test, y_pred, nn=False):
    '''
    Count the number of correct 1s and 0s.
    '''
    y_test = y_test.to_numpy()
    y_pred = y_pred.flatten()
    if nn:
        y_pred = (y_pred>=.5).astype(int)
    counter = list(compress(y_test, np.equal(y_test, y_pred).tolist()))
    zeros = counter.count(0) + counter.count(-1) # count the number of correctly predicted sensitive samples
    ones = counter.count(1) # count the number of correctly predicted resistant samples
    
    return zeros, ones

def neural_net_with_hyperband(
        X_train_ros, y_train_ros, X_val_, y_val_, X_test, y_test,
        step_size_nodes, min_nodes, max_nodes, factor, executions_per_trial,
        patience, min_delta, epochs, learning_rate_min, learning_rate_max,
        max_layers, metadata, output_dir, hyperband_iterations=1, random_state=0,
        plt_confusion=False):
    '''
    Builds hyperband-tuned neural net using keras, tuned on validation ROCAUC.
    epochs is the maximum number of epochs any one model is trained for, and
    factor is the reduction factor for the number of models and epochs per
    hyperband bracket. Nodes per layer are sampled on a log scale, with
    step_size_nodes as the multiplier between consecutive values. The best
    hyperparameters are refit on the randomly oversampled training data with
    early stopping on the validation data. Makes predictions on the testing
    dataset. Plots confusion matrix and ROCAUC plot.
    '''
    def build_model(hp):
        model = keras.Sequential()
        l2 = hp.Float('l2', min_value=1e-6, max_value=1e-2, sampling='log')
        for i in range(hp.Int('n_layers', min_value=1, max_value=max_layers)):
            model.add(Dense(units=hp.Int('units__' + str(i), min_value=min_nodes,
                                         max_value=max_nodes, step=step_size_nodes, sampling='log'),
                            activation='relu', kernel_regularizer=keras.regularizers.L2(l2)))
            model.add(layers.Dropout(hp.Float('dropout__' + str(i), min_value=0.0,
                                              max_value=0.5, step=0.1)))
        model.add(layers.Dense(1, activation='sigmoid'))
        learning_rate = hp.Float('learning_rt', min_value=learning_rate_min,
                                 max_value=learning_rate_max, sampling='log')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        return model

    keras.utils.set_random_seed(random_state)
    y_test = y_test.replace(-1,0)
    y_train_ros = y_train_ros.replace(-1,0)
    y_val_ = y_val_.replace(-1,0)
    build_model(keras_tuner.HyperParameters())

    tuner = keras_tuner.Hyperband(hypermodel=build_model,
                                  objective=keras_tuner.Objective('val_auc', direction='max'),
                                  max_epochs=epochs, factor=factor,
                                  hyperband_iterations=hyperband_iterations,
                                  executions_per_trial=executions_per_trial, seed=random_state,
                                  overwrite=True, directory=output_dir, project_name='nn_hb')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                               patience=patience, min_delta=min_delta)
    # hyperband sets the number of epochs per trial itself, up to max_epochs
    tuner.search(X_train_ros, y_train_ros,
                validation_data=(X_val_, y_val_), callbacks=[stop_early], verbose=0)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print('nn best params:', best_hp.values)

    # the best trial may come from a bracket that stopped after only a few
    # epochs, so refit its hyperparameters for up to the full epoch budget
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(X_train_ros, y_train_ros, validation_data=(X_val_, y_val_), epochs=epochs,
                   callbacks=[keras.callbacks.EarlyStopping(
                       monitor='val_auc', mode='max', patience=patience,
                       min_delta=min_delta, restore_best_weights=True)],
                   verbose=0)

    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['tissue'].unique():
        X_test_tissue = X_test[metadata_test['tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['tissue']==tissue]
        # y_test_tissue = y_test_tissue.replace(-1,0)
        metadata_tissue = metadata_test[metadata_test['tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = best_model.predict(X_test_tissue)
            if has_enough_samples(y_test_tissue['label']):         
                acc, rocauc, fpr, tpr = evaluate_keras(
                    best_model, X_test_tissue, y_test_tissue['label'], y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred, nn=True)
                if plt_confusion:
                    plot_confusion_matrix(
                        y_test_tissue['label'], y_pred, output_dir=output_dir,
                        model_name='nn_hb_' + tissue, nn=True)
                evaluation_df.append(['nn_hb', tissue, acc, rocauc, zeros, ones])
            else:
                acc = best_model.evaluate(X_test_tissue, y_test_tissue, return_dict=True)['accuracy']
                zeros, ones = counter(y_test_tissue['label'], y_pred, nn=True)
                evaluation_df.append(['nn_hb', tissue, acc, np.nan, zeros, ones])
    
    y_pred = best_model.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate_keras(best_model, X_test, y_test['label'], y_pred, 
                                           )
    zeros, ones = counter(y_test['label'], y_pred, nn=True)
    evaluation_df.append(['nn_hb', 'all_tissues', acc, rocauc, zeros, ones])
    
    if plt_confusion:
        plot_confusion_matrix(
            y_test['label'], y_pred, output_dir, model_name='nn_hb', nn=True)
    
    return pd.DataFrame(evaluation_df), y_pred

def random_forest(
        X_train_ros, y_train_ros, X_test, y_test, output_dir,
        metadata, X_val=None, y_val=None, max_estimators=500, eta=3,
        random_state=0, plt_confusion=False):
    '''
    Builds hyperband-tuned random forest model, using the number of trees as
    the hyperband budget. Fits the model to the randomly oversampled training
    data. Makes predictions on the testing dataset. Plots confusion matrix and
    ROCAUC plot.
    '''
    param_distributions = {
        'max_depth': [2, 3, 5, 10, 20, None],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.1, 0.3, 0.5],
        'class_weight': [None, 'balanced'],
        'bootstrap': [True, False]}
    clf, best_params, best_score = hyperband_search(
        RandomForestClassifier(random_state=random_state), param_distributions,
        X_train_ros, y_train_ros, X_val, y_val, resource='n_estimators',
        max_resource=max_estimators, eta=eta, random_state=random_state)
    print('rf best params:', best_params, 'val rocauc:', best_score)

    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['tissue'].unique():
        X_test_tissue = X_test[metadata_test['tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)         
            if has_enough_samples(y_test_tissue['label']):           
                acc, rocauc, fpr, tpr = evaluate(
                    y_test_tissue, y_pred, predict_score(clf, X_test_tissue))
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                if plt_confusion:
                    plot_confusion_matrix(
                        y_test_tissue['label'], y_pred, output_dir,
                        model_name='rf_' + tissue, nn=False)
                evaluation_df.append(['rf', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['rf', tissue, acc, np.nan, zeros, ones])

    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred, predict_score(clf, X_test))
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['rf', 'all_tissues', acc, rocauc, zeros, ones])
    
    if plt_confusion:
        plot_confusion_matrix(
            y_test['label'], y_pred, output_dir, model_name='rf', nn=False)
    
    return pd.DataFrame(evaluation_df), y_pred

def ridge_classifier(
        X_train_ros, y_train_ros, X_test, y_test, output_dir,
        metadata, X_val=None, y_val=None, eta=3, random_state=0, plt_confusion=False):
    '''
    Builds hyperband-tuned ridge classifier model, using the number of training
    samples as the hyperband budget. Fits the model to the randomly oversampled
    training data. Makes predictions on the testing dataset. Plots confusion
    matrix and ROCAUC plot.
    '''
    param_distributions = {
        'alpha': loguniform(1e-4, 1e4),
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']}
    clf, best_params, best_score = hyperband_search(
        RidgeClassifier(), param_distributions, X_train_ros, y_train_ros, X_val, y_val,
        resource='n_samples', eta=eta, random_state=random_state)
    print('ridge best params:', best_params, 'val rocauc:', best_score)

    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)

    evaluation_df = []
    for tissue in metadata_test['tissue'].unique():
        X_test_tissue = X_test[metadata_test['tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)
            if has_enough_samples(y_test_tissue['label']):
                acc, rocauc, fpr, tpr = evaluate(
                    y_test_tissue, y_pred, predict_score(clf, X_test_tissue))
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                if plt_confusion:
                    plot_confusion_matrix(
                        y_test_tissue['label'], y_pred, output_dir,
                        model_name='ridge_classification_' + tissue, nn=False)
                evaluation_df.append(['ridge_classification', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['ridge_classification', tissue, acc, np.nan, zeros, ones])

    y_pred = clf.predict(X_test)

    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred, predict_score(clf, X_test))
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['ridge_classification', 'all_tissues', acc, rocauc, zeros, ones])

    if plt_confusion:
        plot_confusion_matrix(
            y_test['label'], y_pred, output_dir, model_name='ridge_classification', nn=False)

    return pd.DataFrame(evaluation_df), y_pred

def svm_model(
    X_train, y_train, X_test, y_test, output_dir, metadata, X_val=None, y_val=None,
    eta=3, random_state=0):
    '''
    Builds hyperband-tuned svm, using the number of training samples as the
    hyperband budget.
    '''
    param_distributions = {
        'C': loguniform(1e-3, 1e3),
        'gamma': loguniform(1e-5, 1e1),
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced']}
    clf, best_params, best_score = hyperband_search(
        svm.SVC(), param_distributions, X_train, y_train, X_val, y_val,
        resource='n_samples', eta=eta, random_state=random_state)
    print('svm best params:', best_params, 'val rocauc:', best_score)

    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)

    evaluation_df = []
    for tissue in metadata_test['tissue'].unique():
        X_test_tissue = X_test[metadata_test['tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['tissue']==tissue]
        metadata_tissue = metadata_test[metadata_test['tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)
            if has_enough_samples(y_test_tissue['label']):
                acc, rocauc, fpr, tpr = evaluate(
                    y_test_tissue, y_pred, predict_score(clf, X_test_tissue))
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                # if plt_confusion:
                #     plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir,
                #                           feature_selection,
                #                         model_name='elastic_net_' + tissue, nn=False)
                # if plt_rocauc:
                #     plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='ridge_'
                #                 + tissue)
                evaluation_df.append(['svm', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['svm', tissue, acc, np.nan, zeros, ones])

    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred, predict_score(clf, X_test))
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['svm', 'all_tissues', acc, rocauc, zeros, ones])

    return pd.DataFrame(evaluation_df), y_pred

def l1_svm(
        X_train, y_train, X_test, y_test, output_dir, metadata, X_val=None, y_val=None,
        eta=3, random_state=0, plt_confusion=False):
    '''
    Builds hyperband-tuned linear svm with an L1 penalty and squared hinge loss,
    using the number of training samples as the hyperband budget. The L1
    penalty sets most feature weights to zero, so the model also does feature
    selection; the number of features kept is printed.
    '''
    param_distributions = {
        'C': loguniform(1e-3, 1e2),
        'class_weight': [None, 'balanced']}
    clf, best_params, best_score = hyperband_search(
        svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, max_iter=5000),
        param_distributions, X_train, y_train, X_val, y_val,
        resource='n_samples', eta=eta, random_state=random_state)
    print('svm_l1 best params:', best_params, 'val rocauc:', best_score,
          'nonzero features:', int(np.count_nonzero(clf.coef_)), '/', clf.coef_.shape[1])

    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)

    evaluation_df = []
    for tissue in metadata_test['tissue'].unique():
        X_test_tissue = X_test[metadata_test['tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['tissue']==tissue]
        if X_test_tissue.shape[0]>0:
            y_pred = clf.predict(X_test_tissue)
            if has_enough_samples(y_test_tissue['label']):
                acc, rocauc, fpr, tpr = evaluate(
                    y_test_tissue, y_pred, predict_score(clf, X_test_tissue))
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                if plt_confusion:
                    plot_confusion_matrix(
                        y_test_tissue['label'], y_pred, output_dir,
                        model_name='svm_l1_' + tissue, nn=False)
                evaluation_df.append(['svm_l1', tissue, acc, rocauc, zeros, ones])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                zeros, ones = counter(y_test_tissue['label'], y_pred)
                evaluation_df.append(['svm_l1', tissue, acc, np.nan, zeros, ones])

    y_pred = clf.predict(X_test)

    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred, predict_score(clf, X_test))
    zeros, ones = counter(y_test['label'], y_pred)
    evaluation_df.append(['svm_l1', 'all_tissues', acc, rocauc, zeros, ones])

    if plt_confusion:
        plot_confusion_matrix(
            y_test['label'], y_pred, output_dir, model_name='svm_l1', nn=False)

    return pd.DataFrame(evaluation_df), y_pred