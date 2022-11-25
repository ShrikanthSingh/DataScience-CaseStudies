import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def split_train_test(x_data, y_data):
    """
    Function to split the training and testing data
    Args:
        x_data (DataFrame): Features 
        y_data (DataFrame): Target labels
    Returns:
        X_train (DataFrame): Training data features
        X_test (DataFrame): Test data features
        y_train (DataFrame): Training data labels
        y_test (DataFrame): Test data labels 
    """
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                        shuffle= True, 
                                                        test_size=0.4, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


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


