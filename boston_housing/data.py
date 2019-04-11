import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def distmap(df):
    """
    distmap creates a matrix of boxplots of a given dataframe.
    :param df: dataframe
    :return: scatterplot of the distributions of specified variables
    """
    check = [['crim', 'zn'],
             ['indus', 'nox'],
             ['rm', 'age'],
             ['dis', 'tax'],
             ['ptratio', 'lstat']]

    fig, ax = plt.subplots(5, 2, figsize=(10, 9))
    plt.subplots_adjust(hspace=0.40)
    for i in range(5):
        for j in range(2):
            ax[i, j].hist(df[check[i][j]], bins=30)
            ax[i, j].set_title(check[i][j])

    return 0

def scaler(df):
    """
    scaler will scale the features of a given dataframe
    :param df: dataframe
    :return: scaled version of df
    """
    scaler = preprocessing.StandardScaler()
    unscaled_feats = df.drop(['chas', 'medv'], axis=1)
    names = unscaled_feats.columns

    scaled_feats = scaler.fit_transform(unscaled_feats)
    scaled_feats = pd.DataFrame(scaled_feats, columns=names)
    scaled_feats = pd.concat([scaled_feats, df['medv']], axis=1)

    return scaled_feats

def corr_sort(df):
    """

    :param df: dataframe
    :return: return a matrix of sorted correlations
    """
    abs_corr = df.corr().abs()
    unstacked_corr = abs_corr.unstack()
    sorted_corr = unstacked_corr.sort_values(kind='quicksort', ascending=False)
    return sorted_corr

def get_redundant_pairs(df):
    """

    :param df: dataframe
    :return: a list of redundant pairs that will later be dropped
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    """

    :param df: dataframe
    :param n: number of pairs to show, default 5 pairs
    :return: a list of the top correlations
    """
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def tol_vif_table(df, n = 5):
    """

    :param df: dataframe
    :param n: number of pairs to show
    :return: table of correlations, tolerances, and VIF
    """
    cor = get_top_abs_correlations(df, n)
    tol = 1 - cor ** 2
    vif = 1 / tol
    cor_table = pd.concat([cor, tol, vif], axis=1)
    cor_table.columns = ['Correlation', 'Tolerance', 'VIF']
    return cor_table

def train_test_splits(df):
    """

    :param df: dataframe
    :return: train, test splits for both predictors and target variables
    """
    X = df.drop('medv', axis=1)
    y = df['medv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)
    return X_train, X_test, y_train, y_test

def get_lasso_reg():
    lasso = Lasso()
    lasso01 = Lasso(alpha=0.1, max_iter=10e5)
    lasso001 = Lasso(alpha=0.01, max_iter=10e5)
    lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
    return lasso, lasso01, lasso001, lasso00001

def lasso_reduction(df):
    lasso, lasso01, lasso001, lasso00001 = get_lasso_reg()
    X_train, X_test, y_train, y_test = train_test_splits(df)

    # Fit Lasso Regressions
    lasso.fit(X_train, y_train)
    lasso01.fit(X_train, y_train)
    lasso001.fit(X_train, y_train)
    lasso00001.fit(X_train, y_train)

    used_feats = pd.Series(
        {'1': np.sum(lasso.coef_ != 0), '0.1': np.sum(lasso01.coef_ != 0), '0.01': np.sum(lasso001.coef_ != 0),
         '0.001': np.sum(lasso00001.coef_ != 0)})
    train_scores = pd.Series({'1': lasso.score(X_train, y_train), '0.1': lasso01.score(X_train, y_train),
                              '0.01': lasso001.score(X_train, y_train), '0.001': lasso00001.score(X_train, y_train)})
    test_scores = pd.Series(
        {'1': lasso.score(X_test, y_test), '0.1': lasso01.score(X_test, y_test), '0.01': lasso001.score(X_test, y_test),
         '0.001': lasso00001.score(X_test, y_test)})

    evaluation_table = pd.DataFrame(
        {'Used Features': used_feats, 'Training Score': train_scores, 'Test Score': test_scores})

    return evaluation_table

def get_lasso_coef(df, a):
    lasso01 = Lasso(alpha=a, max_iter=10e5)
    X_train, X_test, y_train, y_test = train_test_splits(df)
    lasso01.fit(X_train, y_train)
    lasso01_feats = X_train.columns[lasso01.coef_ != 0]

    return lasso01_feats

