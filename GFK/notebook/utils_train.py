import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hyperopt import fmin
from xgboost import XGBClassifier
from hyperopt import hp, tpe, Trials, STATUS_OK

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


def impute_column(df, input_column, info_column):
    column_impute = df[input_column]\
                    .fillna(df.groupby(info_column)[input_column].transform('mean'))
    return column_impute


def org_results(trials, hyperparams, model_name):
    fit_idx = -1
    for idx, fit  in enumerate(trials):
        hyp = fit['misc']['vals']
        xgb_hyp = {key:[val] for key, val in hyperparams.items()}
        if hyp == xgb_hyp:
            fit_idx = idx
            break
            
    train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])
    acc = round(trials[fit_idx]['result']['accuracy'], 3)
    train_auc = round(trials[fit_idx]['result']['train auc'], 3)
    test_auc = round(trials[fit_idx]['result']['test auc'], 3)

    results = {
        'model': model_name,
        'parameter search time': train_time,
        'accuracy': acc,
        'test auc score': test_auc,
        'training auc score': train_auc,
        'parameters': hyperparams
    }
    return results


def xgb_objective(space, early_stopping_rounds=50):
    
    with open(os.path.join('..', 'data', 'data_segments.pkl'), 'rb') as f:
        data_segments = pickle.load(f)
    
    train_x = data_segments['train_x']
    train_y = data_segments['train_y'] 
    test_x = data_segments['test_x'] 
    test_y = data_segments['test_y']

    model = XGBClassifier(
        learning_rate = space['learning_rate'], 
        n_estimators = int(space['n_estimators']), 
        max_depth = int(space['max_depth']), 
        min_child_weight = space['m_child_weight'], 
        gamma = space['gamma'], 
        subsample = space['subsample'], 
        colsample_bytree = space['colsample_bytree'],
        objective = 'binary:logistic',
        use_label_encoder=False
    )

    model.fit(train_x, train_y, 
              eval_set = [(train_x, train_y), (test_x, test_y)],
              eval_metric = 'auc',
              early_stopping_rounds = early_stopping_rounds,
              verbose = 0)
     
    predictions = model.predict(test_x)
    test_preds = model.predict_proba(test_x)[:,1]
    train_preds = model.predict_proba(train_x)[:,1]
    
    xgb_booster = model.get_booster()
    train_auc = roc_auc_score(train_y, train_preds)
    test_auc = roc_auc_score(test_y, test_preds)
    accuracy = accuracy_score(test_y, predictions) 

    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,
            'test auc': test_auc, 'train auc': train_auc
           }

def generate_performance_metrics(test_labels, prediction_labels): 
    """
    Function to compute the classification performance metrics
    Args:
        test_labels (numpy.array): Array of actual test labels
        prediction_labels (numpy.array): Array of predicted test labels
    Returns:
        df_metrics (DataFrame): Classification report
    """
    acc_score = round(accuracy_score(test_labels, prediction_labels), 3) 
    print(f'Accuracy of the model = {round(acc_score*100, 2)}')

    df_metrics = pd.DataFrame(classification_report(prediction_labels, 
                                                    test_labels,
                                                    output_dict=True)).T

    df_metrics['support'] = df_metrics.support.apply(int)

    cm = confusion_matrix(prediction_labels, test_labels)
    cm_df = pd.DataFrame(cm,
                        index =  list(np.unique(test_labels)),
                        columns = list(np.unique(test_labels)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap='Blues')
    sns.set(font_scale=2.0)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

    return np.round(df_metrics, decimals=2)