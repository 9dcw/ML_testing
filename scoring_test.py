import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble.partial_dependence import plot_partial_dependence
#from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score

#from sklearn.ensemble.partial_dependence import partial_dependence
import sys
from sklearn import preprocessing

def main():

    path = 'c:\\users\\dwright\\code\\'
    filename = path + 'scorebase.csv'
    readRows = 1000

    rnd = np.random.rand(readRows, 1) ** 3
    biny = np.round(rnd, 0) # binary y
    y = biny * np.random.lognormal(10, 2, (readRows, 1))
    y = y.flatten().astype('int')

    raw_train_data = pd.read_csv(filename, nrows=readRows)

    featureNames = ['protection_class', 'numberoffamilies', 'yearbuilt',
                    'numberofbathrooms', 'construction', 'supplementalheatingsource',
                    'pool', 'wiringmaterial']

    featureNames = [i for i in raw_train_data.columns if i.lower() in featureNames]
    print 'feature names', featureNames
    print 'raw data shape', raw_train_data.shape[0]
    train_data = raw_train_data.loc[:, featureNames]

    kf = KFold(train_data.shape[0], n_folds=2)
    train_data = preprocess(train_data)

    print train_data.shape

    for trainI, testI in kf:

        X_train = train_data.ix[trainI, :]
        X_test = train_data.ix[testI, :]
        print 'train shape:', X_train.shape
        print 'test shape:', X_test.shape

        y_train = y[trainI]
        y_test = y[testI]

        lin = LinearRegression()
        lin.fit(X_train, y_train)

        y_pred = lin.predict(X_test)
        r2_lin = r2_score(y_test, y_pred)
        clf = GradientBoostingRegressor()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print 'inear score:', r2_lin, 'boost score:', r2

    randforest = RandomForestRegressor()

    # let's fit a random forest!
    randforest.fit(X_train, y)
    # ok now what??!?!?

    # would be useful to find the most important variables and plot those against loss cost
    # bow do we do feature pruning with random forests?

    # then I want to chart the various accuracy measures
    # then I try another model and try that.

    # I also want to look into seaborn and see what there is in there that I can use
    # I'm going to want to learn how lasso can work with binarized features
    # and comb through the various regression models and learn the math
    # and work through the parameter search!

    # features = range(len(featureNames))
    # partial dependence isn't going to work here because we are encoding the labels...
    # I will need to figure out how to do the binary label encoding better!

    #fig, axs = plot_partial_dependence(clf, X_train, features=features, feature_names=featureNames,
    #                                   n_jobs=3, grid_resolution=50)

    # how about we do some grid search plotting!
    # or just a simple linear regression on the features
    # run it!
    # plt.show()

    # then the relative predictive power


def preprocess(train_data):

    typeCols = zip(train_data.dtypes, train_data.columns)

    for item in typeCols:
        tp = item[0]
        colName = item[1]

        # I'd like the min, max, mean, mode and median of each item
        print tp, colName #, train_data[colName][:10]

        if colName == 'Protection_Class':
            #print 'fixing', colName
            train_data.loc[train_data[colName] == '8B', colName] = '8'

            #print 'fixed', colName
            train_data = binarize(train_data, colName)
        if colName == 'NumberOfBathrooms':

            mn = train_data.loc[train_data[colName] != 'Unknown', colName].astype('float').mean()

            train_data.loc[train_data[colName] == 'Unknown', colName] = mn
            train_data.loc[:, colName] = train_data[colName].astype('float')

        if colName.lower() == 'construction':
            train_data.loc[train_data[colName] == 'Framing, Wood', colName] = 'Frame'
            train_data.loc[train_data[colName] == 'Superior - Non '
                                                  'Combustible or Fire Resistive',
                                                  colName] = 'Superior'
            train_data = binarize(train_data, colName)
        if colName == 'SupplementalHeatingSource':

            train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode().values[0]
            train_data = binarize(train_data, colName)

        if colName == 'Pool':
            m = train_data[colName].mode().values[0]

            train_data.fillna(m)
            train_data.loc[train_data[colName] == 'Unknown', colName] = m
            train_data.loc[train_data[colName] == '8', colName] = m

            #x = train_data[colName].unique()

            train_data = binarize(train_data, colName)

        if colName == 'WiringMaterial':

            train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode().values[0]
            c = 0
            # go through the list of materials
            for forName in train_data[colName].value_counts().index:
                # these are the index names of the rows
                ct = train_data[colName].value_counts()[forName]
                # if a material with a small number of instances, make it 'other'
                if ct < train_data.shape[0] * .05:
                    print ct, forName
                    train_data.loc[train_data[colName] == forName, colName] = 'Other'

                c += 1
            #print train_data[colName].value_counts()

            # all looks like romex and BX.. is that right?
            train_data = binarize(train_data, colName)
    return train_data


def binarize(train_data, colName):
    enc = preprocessing.LabelBinarizer()
    print 'encoding', colName

    # fit the encoder
    enc.fit(train_data[colName])

    # build the encodings
    train_enc = enc.transform(train_data[colName])

    # zip up and dict the column names, which will be 'old: col_value'
    l = zip(range(1, len(enc.classes_)+1), enc.classes_)
    renameDict = {i[0]: colName + '_' + str(i[1]) for i in l}

    # rename the new columns and build a dataframe
    train_enc = pd.DataFrame(train_enc).rename(columns=renameDict)

    # drop the old values
    train_data = train_data.drop(colName, 1)

    # concatenate in the new columns
    train_data = pd.concat((train_data, pd.DataFrame(train_enc)), 1)
    print 'encoded', colName
    return train_data

if __name__ == '__main__':
    main()