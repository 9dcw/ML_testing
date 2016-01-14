print 'loading libraries'
import pandas as pd
import os
import numpy as np
import sys
# import csv
# import matplotlib as mlp
import matplotlib.pyplot as plt
# import sklearn

from sklearn import cross_validation
# from sklearn import tree
from sklearn import svm
# from sklearn import ensemble
# from sklearn import neighbors
from sklearn import feature_extraction
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

plt.style.use('fivethirtyeight')
_PLT_LEGEND_OPTIONS = dict(loc="upper center",
                           bbox_to_anchor=(0.5, -0.15),
                           fancybox=True,
                           shadow=True,
                           ncol=3)

print 'loaded libraries'
# M-b, M-f

def main():

    path = '/users/davecwright/documents/kaggle/liberty_fire_cost/'

    train_name = path + 'train.csv'
    test_name = path + 'test.csv'
    # f = open(path + 'train.csv', 'rb')

    readRows = 2000 #None for all
    print 'loading train_data'
    train_data = pd.read_csv(train_name, nrows=readRows)

    print 'train_data loaded'
    #print train_data.head()
    #print train_data.dtypes
    #print train_data.columns

    y_data = train_data['target'].values
    y_data = y_data.astype(float)


    fig = plt.figure(figsize=(12, 9))
    #ax = fig.add_subplot(111)

    #ax.plot(np.sort(y_data))
    #plt.xlabel("Number of Features")
    #plt.ylabel("claims cost")
    #plt.title("claims_cost")
    #ax.set_xscale("log")

    #ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])
    #ax.legend(**_PLT_LEGEND_OPTIONS)
    #plt.show()


    train_data = train_data.drop('target', 1)

    # A - preprocessing
    # encode the text variables, var1-var9, Z values are NaN

    # fill in missing values in the text variables and in the continuous variables
    # skip continuous for now

    # A1. make everything numeric

    train_data = encode_impute(train_data)
    # A2. standardize the feature scales

    scaler = StandardScaler()
    scaler.fit(train_data)
    X_train = scaler.transform(train_data)

    # A3. build new features through the interactions of various items
    # skip for now

    #  A4. dimensionality reduction to take the feature set back down to something more manageable.

    est_clf = svm.SVR(kernel='linear', C=1)
    rks = select_ests(X_train, y_data, 100, est_clf)
    X_train = X_train[:, rks]

    clf = svm.SVR(kernel='linear', C=1)
    acy = cv(X_train, y_data, clf, None, estimator_name(clf))

    # the point here is to understand what accuracy is

    print 'accuracy:', acy

    #select_model(X_train, y_data)

    # B. split out test and fit sets

    test_data = pd.read_csv(test_name, nrows=readRows)
    test_data = encode_impute(test_data)
    X_test = scaler.transform(test_data)

    # C. pick best parameter set for various models

    # here I need to plot out the data and the results... maybe just a quick plot of the results first?
    # I don't know what accuracy or scoring mean..

    print 'done?'

    # should be doing the parameter search next... I'm skipping that and moving to the model..


def estimator_name(clf):
    return type(clf).__name__


def select_ests(X, y, nfeats, clf):
    rfe = feature_selection.RFE(estimator=clf, n_features_to_select=100, step=10)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    # 1 values refer to the features that are taken
    rks = ranking[ranking == 1]

    return rks


def cv(X, y, clf, nfeats, clfname, scoring=metrics.r2_score, n_folds=10):

    # returns an index for the train and test sets

    # stratified kfold is a classification thing that keeps the proportions similar for
    # the different y values.. I need just straight KFold

    stratified_k_fold = cross_validation.KFold(y, n_folds=n_folds)
    accuracy, ii = 0., 0
    for train, test in stratified_k_fold:
        ii += 1
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        score = scoring(y_test, y_pred)
        print clfname, ii, 'r-squared:', score

        accuracy += score
    accuracy /= float(n_folds)
    return accuracy


def plot_accuracies(accuracies, xvals, legends):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    for ii in range(0, accuracies.shape[0]):
        ax.plot(xvals, accuracies[ii, :], color=next(colors), marker=next(markers), label=legends[ii])
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Features")
    ax.set_xscale("log")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])
    ax.legend(**_PLT_LEGEND_OPTIONS)
    plt.show()


def select_model(X, y, scoring = metrics.accuracy_score):
    # here I need to find the regressions and list them out
    # then parametarize them...

    # fuck it I'll just run the parameter search myself... not like I know what they are anyway...

    #n_features = np.array([10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000])
    clfs = [

        linear_model.Ridge(),
        linear_model.LinearRegression(),
        linear_model.ElasticNet(),
        linear_model.Lasso(),
      ]

    regressor_names = map(estimator_name, clfs)

    feature_selection_methods = [feature_selection.f_classif]
    accuracies = np.zeros((len(clfs), len(n_features), len(feature_selection_methods)))
    for kk in range(len(feature_selection_methods)):
        X_feature_selected = X.copy()#.toarray()
        for jj in range(len(n_features)):
            for ii in range(len(clfs)):
                accuracies[ii, jj, kk] = cv(X_feature_selected, y, clfs[ii],
                                            regressor_names[ii], scoring=scoring)
    for k in range(len(feature_selection_methods)):
        for i in range(len(clfs)):
            print "%22s " % regressor_names[i],
            for j in range(accuracies.shape[1]):
                print "%5.3f" % accuracies[i, j, k],
            print plot_accuracies(accuracies[:, :, k], n_features, regressor_names)



    return

def encode_impute(train_data):
    enc = preprocessing.LabelEncoder()
    l = zip(train_data.dtypes, train_data.columns)
    for i in l:

        tp = i[0]
        nm = i[1]

        if tp == 'object':
            # get the name of the column
            # print i
            #print 'col name: ', nm, tp
            # print 'grouping and count for this column', train_data.groupby(nm).size()

            # here we need to decide what to do with the rows that have Z values.
            # we could use the imputer
            # transform the train_data

            # this creates many columns from one. We need to deal with that by adding
            # them in gracefully... append at the end?
            enc.fit(train_data[nm])
            train_enc = enc.transform(train_data[nm])
            train_data = train_data.drop(nm, 1)
            train_data = pd.concat((train_data, pd.DataFrame(train_enc)), 1)

        else:
            nans = len(train_data[nm]) - train_data[nm].count()
            #print i, train_data[nm].min(), train_data[nm].mean(), train_data[nm].max(), nans

            imp = Imputer(missing_values='NaN', strategy='median', axis=0)

            train_data[nm] = imp.fit_transform(pd.DataFrame(train_data[nm]))

    print 'test train_data shape', train_data.shape
    return train_data


def drop_nans(df):
    x = 0
    s = df.shape
    for tp in df.dtypes:
        print tp, x, df.iloc[:, x].describe()
        if tp == 'object':
            df = df.iloc[:, x]
            df = df[df != 'Z']
            train_data.iloc[:, x] = df[df != 'NA']
            print 'nan values:', len(df.iloc[:, x]) - df.count()
        x += 1

    df = df.dropna(how='any')
    print 'old:', s, 'new:', df.shape

    return df

if __name__ == '__main__':
    main()