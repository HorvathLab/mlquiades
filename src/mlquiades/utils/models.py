import keras
import keras_tuner
import numpy as np
import pandas as pd
from .plotting import *
from keras import layers
from sklearn import metrics, tree
from tensorflow.keras.models import Sequential
from sklearn.linear_model import RidgeClassifier
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve, roc_auc_score

def evaluate(
        y_test, y_pred):
    '''
    Evaluates y-label predictions for testing dataset. Returns accuracy, false
    positive rate, true positive rate and ROCAUC values. This is only for non-keras
    models.
    '''
    acc = metrics.accuracy_score(y_test, y_pred)
    # f1 = f1_score(np.array(y_test), (y_pred).astype(int))    
    fpr,tpr,thresholds = roc_curve(y_test,y_pred)
    rocauc = roc_auc_score(y_test,y_pred)
    
    return acc, rocauc.item(), fpr, tpr

def evaluate_keras(
        model, X_test, y_test, y_pred):
    '''
    Evaluates y-label predictions for testing dataset. Returns accuracy, false
    positive rate, true positive rate and ROCAUC values. This is only for keras
    models.
    '''
    loss, acc = model.evaluate(X_test, y_test)
    fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
    rocauc = auc(fpr, tpr)
    # f1_metric = keras.metrics.F1Score(threshold=0.5)
    # f1_metric.update_state(np.array([y_test.astype(np.float32)]).T,y_pred)
    
    return acc, rocauc, fpr, tpr

def decision_tree(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection, metadata,
        plt_confusion=True, plt_rocauc=True):
    '''
    Builds decision tree model. Fits the model to the randomly oversampled
    training data. Makes predictions on the testing dataset. Plots confusion
    matrix and ROCAUC plot.
    '''
    clf = tree.DecisionTreeClassifier(max_depth=2)
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
                if plt_confusion:
                    plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                                        model_name='dt_' + tissue, nn=False)
                if plt_rocauc:
                    plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='dt_' + tissue)
                evaluation_df.append(['dt', tissue, acc, rocauc])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                evaluation_df.append(['dt', tissue, acc, 0])
    
    y_pred = clf.predict(X_test)
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred)
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                            model_name='dt', nn=False)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='dt')
    
    evaluation_df.append(['dt', 'all_tissues', acc, rocauc])
    
    return pd.DataFrame(evaluation_df)

def gradient_boosted_decision_tree(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection,
        metadata, plt_confusion=True, plt_rocauc=True):
    '''
    Builds gradient-boosted decision tree model. Fits the model to the randomly
    oversampled training data. Makes predictions on the testing dataset. Plots
    confusion matrix and ROCAUC plot.
    '''
    clf = GradientBoostingClassifier(n_estimators=2, learning_rate=1.5,
                                     max_depth=2,random_state=0).fit(X_train_ros, y_train_ros)
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
                if plt_confusion:
                    plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                                        model_name='gbdt_' + tissue, nn=False)
                if plt_rocauc:
                    plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='gbdt_' + tissue)
                evaluation_df.append(['gbdt', tissue, acc, rocauc])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                evaluation_df.append(['gbdt', tissue, acc, 0])
    
    y_pred = clf.predict(X_test)
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred)
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection, 
                              model_name='gbdt', nn=False)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='gbdt')
    
    evaluation_df.append(['gbdt', 'all_tissues', acc, rocauc])

    return pd.DataFrame(evaluation_df)

def neural_net(
        X_train_ros, y_train_ros, X_val_, y_val_, X_test, y_test, output_dir,
        feature_selection, metadata, plt_confusion=True, plt_rocauc=True):
    '''
    Builds neural net using keras. Fits the model to the randomly oversampled
    training data. Makes predictions on the testing dataset. Plots confusion
    matrix and ROCAUC plot.
    '''
    model = Sequential()
    model.add(Dense(12, input_shape=(X_train_ros.shape[1],), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                             min_delta=.01)
    optimiizer = keras.optimizers.Adam(learning_rate=.0001)
    
    model.compile(loss='binary_crossentropy', optimizer=optimiizer, metrics=['accuracy'])
    y_train_ros = y_train_ros.replace(-1,0)
    y_val_ = y_val_.replace(-1,0)
    y_test = y_test.replace(-1,0)
    model.fit(X_train_ros, y_train_ros, validation_data = (X_val_, y_val_),
              epochs=100, batch_size=50, callbacks=[callback], verbose=0)
    
    metadata_test = metadata[metadata['train_val_test']=='test'].reset_index(drop=True)
    
    evaluation_df = []
    for tissue in metadata_test['Tissue'].unique():
        X_test_tissue = X_test[metadata_test['Tissue']==tissue]
        y_test_tissue = y_test.reset_index(drop=True)[metadata_test['Tissue']==tissue].reset_index(drop=True)
        y_test_tissue = y_test_tissue.replace(-1,0)
        metadata_tissue = metadata_test[metadata_test['Tissue']==tissue]
        if X_test_tissue.shape[0]>0:          
            y_pred = model.predict(X_test_tissue)         
            if len(y_test_tissue['label'].unique())>1:   
                acc, rocauc, fpr, tpr = evaluate_keras(model, X_test_tissue, y_test_tissue['label'], y_pred)
                if plt_confusion:
                    plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                                        model_name='nn_' + tissue, nn=False)
                if plt_rocauc:
                    plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn_' + tissue)
                evaluation_df.append(['nn', tissue, acc, rocauc])
            else:
                loss, acc = model.evaluate(X_test_tissue, y_test_tissue)
                evaluation_df.append(['nn', tissue, acc, 0])
    
    y_pred = model.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate_keras(model, X_test, y_test['label'], y_pred)
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                              model_name='nn', nn=True)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn')
    
    evaluation_df.append(['nn', 'all_tissues', acc, rocauc])
    
    return pd.DataFrame(evaluation_df)

def neural_net_with_hyperband(
        X_train_ros, y_train_ros, X_val_, y_val_, X_test, y_test, data_dir,
        step_size_nodes, min_nodes, max_nodes, max_trials, executions_per_trial,
        patience, min_delta, epochs, learning_rate_min, learning_rate_max, output_dir,
        feature_selection, metadata, plt_confusion=True, plt_rocauc=True):
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
                                     executions_per_trial=executions_per_trial, overwrite=True, directory=data_dir)
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
                acc, rocauc, fpr, tpr = evaluate_keras(best_model, X_test_tissue, y_test_tissue['label'], y_pred)
                if plt_confusion:
                    plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                                        model_name='nn_hb_' + tissue, nn=False)
                if plt_rocauc:
                    plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn_hb_' + tissue)
                evaluation_df.append(['nn_hb', tissue, acc, rocauc])
            else:
                loss, acc = best_model.evaluate(X_test_tissue, y_test_tissue)
                evaluation_df.append(['nn_hb', tissue, acc, 0])
    
    y_pred = best_model.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate_keras(best_model, X_test, y_test['label'], y_pred)
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                            model_name='nn_hb', nn=True)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='nn_hb')
    
    evaluation_df.append(['nn_hb', 'all_tissues', acc, rocauc])
    
    return pd.DataFrame(evaluation_df)

def random_forest(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection,
        metadata, max_depth=2, random_state=0, plt_confusion=True, plt_rocauc=True):
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
                if plt_confusion:
                    plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                                        model_name='rf_' + tissue, nn=False)
                if plt_rocauc:
                    plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='rf_' + tissue)
                evaluation_df.append(['rf', tissue, acc, rocauc])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                evaluation_df.append(['rf', tissue, acc, 0])

    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred)
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                              model_name='rf', nn=False)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='rf')
    
    evaluation_df.append(['rf', 'all_tissues', acc, rocauc])
    
    return pd.DataFrame(evaluation_df)

def ridge_classifier(
        X_train_ros, y_train_ros, X_test, y_test, output_dir, feature_selection,
        metadata, plt_confusion=True, plt_rocauc=True):
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
                if plt_confusion:
                    plot_confusion_matrix(y_test_tissue['label'], y_pred, output_dir, feature_selection,
                                        model_name='ridge_' + tissue, nn=False)
                if plt_rocauc:
                    plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='ridge_' + tissue)
                evaluation_df.append(['ridge', tissue, acc, rocauc])
            else:
                acc = metrics.accuracy_score(y_test_tissue, y_pred)
                evaluation_df.append(['ridge', tissue, acc, 0])
    
    y_pred = clf.predict(X_test)
    
    acc, rocauc, fpr, tpr = evaluate(y_test, y_pred)
    if plt_confusion:
        plot_confusion_matrix(y_test['label'], y_pred, output_dir, feature_selection,
                              model_name='ridge', nn=False)
    if plt_rocauc:
        plot_rocauc(fpr, tpr, output_dir, feature_selection, model_name='ridge')
    
    evaluation_df.append(['ridge', 'all_tissues', acc, rocauc])
    
    return pd.DataFrame(evaluation_df)
