import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function to identify the columns containing null or nan values
    Args:
        df (DataFrame): Input dataframe to find the null values
    Returns:
        mis_val_table_ren_columns (DataFrame): Column name and number of 
        null values in it
    '''
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

def make_discrete_plot(df: pd.DataFrame, x_feature: str, y_feature: str):
    """
    Plotting box and strip plot of categorical variable of features against the target variable
    """
    fig = plt.figure(figsize=(20,8))
    gs = GridSpec(1,2)
    sns.boxplot(y=df[y_feature], x=df[x_feature], ax=fig.add_subplot(gs[0,0]))
    plt.ylabel(y_feature, fontsize=16)
    plt.xlabel(x_feature, fontsize=16)
    sns.stripplot(y=df[y_feature], x=df[x_feature], ax=fig.add_subplot(gs[0,1]))
    plt.ylabel(y_feature, fontsize=16)
    plt.xlabel(x_feature, fontsize=16)
    fig.show()

def make_continuous_plot(df: pd.DataFrame, x_feature: str, y_feature: str):
    """
    Plotting scatter plot and its boxcox (1+x) transformation for a given continuous variable 
    feature against the target variable.
    Args:
        df (DataFrame) : Dataframe containing the columns for plotting the continuous variables
        x_feature (str) : Continuous variable on X-axis
        y_feature (str) : Continuous variable on Y-axis

    """
    
    fig = plt.figure(figsize=(20,12))
    fig.subplots_adjust(hspace=.3)
    gs = GridSpec(2,2)
    
    j = sns.scatterplot(y=df[y_feature], 
                        x=boxcox1p(df[x_feature], 0.15), ax=fig.add_subplot(gs[1,1]), palette = 'blue')

    plt.title('BoxCox 0.15\n' + 'Corr: ' + str(np.round(df[y_feature].corr(boxcox1p(df[x_feature], 0.15)),2)) +
              ', Skew: ' + str(np.round(stats.skew(boxcox1p(df[x_feature], 0.15), nan_policy='omit'),2)))
    
    j = sns.scatterplot(y=df[y_feature], 
                        x=boxcox1p(df[x_feature], 0.25), ax=fig.add_subplot(gs[1,0]), palette = 'blue')

    plt.title('BoxCox 0.25\n' + 'Corr: ' + str(np.round(df[y_feature].corr(boxcox1p(df[x_feature], 0.25)),2)) +
              ', Skew: ' + str(np.round(stats.skew(boxcox1p(df[x_feature], 0.25), nan_policy='omit'),2)))
    
    j = sns.distplot(df[x_feature], ax=fig.add_subplot(gs[0,1]), color = 'green')

    plt.title('Distribution\n')
    
    j = sns.scatterplot(y=df[y_feature], 
                        x=df[x_feature], ax=fig.add_subplot(gs[0,0]), color = 'red')

    plt.title('Linear\n' + 'Corr: ' + str(np.round(df[y_feature].corr(df[x_feature]),2)) + ', Skew: ' + 
               str(np.round(stats.skew(df[x_feature], nan_policy='omit'),2)))


def plot_corr_matrix(df: pd.DataFrame):
    """
    Function to plot the heatmap with correlation between the columns
    Args:
        df (DataFrame) : Input dataframe containing columns for computing the correlation
    """
    sns.set(style="darkgrid", font_scale=1.4)
    corr = df.corr()
    f, ax = plt.subplots(figsize=(15, 20))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.title('Correlation Matrix', fontsize=18)
    sns.heatmap(corr, cmap=cmap, vmax=.5, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, 
                annot=True, annot_kws={"fontsize":20})
    plt.show()


def calculate_pear_coeff(df: pd.DataFrame, x_feature: str, y_feature: str):
    """
    Function to calculate the pearson coeffiecient
    df (DataFrame): Input dataframe
    x_feature (str), y_feature (str): Features to compute the correlation between 
    """
    pearson_corr, _ = stats.pearsonr(df[x_feature], df[y_feature])
    print(f'Pearsons correlation between {x_feature} and {y_feature}: %.3f' % pearson_corr)

def make_reg_plot(df: pd.DataFrame, x_feature: str, y_feature: str):
    """
    Function to plot a regression fit line between the given features
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,8))
    axes.plot(x_feature, y_feature, data=df, color='0.75', linestyle=(0, (1, 10)))
    axes.set_title(f"Regression plot between {x_feature} and {y_feature} ", fontsize=20)
    axes = sns.regplot(x=x_feature, 
                       y=y_feature, 
                       data=df, 
                       ax = axes)