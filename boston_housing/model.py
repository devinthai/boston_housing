import logging
from sklearn.linear_model import LinearRegression
from boston_housing import data
import pandas as pd

def run_linear_regression(df,feats):
    lm = LinearRegression()
    X_train, X_test, y_train, y_test = data.train_test_splits(df)

    lasso_X_train = X_train[feats]
    lasso_X_test = X_test[feats]

    lin_reg = lm.fit(lasso_X_train, y_train)

    accuracy = lin_reg.score(lasso_X_test, y_test)

    ctable = coef_table(lin_reg,lasso_X_train)

    print('The prediction accuracy with the linear regression model is {:.1f}%'.format(accuracy * 100))

    return lin_reg, ctable

def coef_table(lm, df):
    coef = lm.coef_
    X_col = df.columns

    table = pd.Series(data = coef, index = X_col)
    table = pd.DataFrame({'Coefficients':table})

    return table

def coef_std_table(ctable, df, feats, dummies):
    norm_feats = df.drop('medv', axis=1)
    chas_medv = df.drop(norm_feats.columns, axis=1)

    norm_feats = pd.concat([norm_feats, chas_medv, dummies], axis=1)
    stds = pd.DataFrame({'Standard Deviations': norm_feats[feats].std()})
    return pd.concat([ctable, stds], axis=1, sort=True)