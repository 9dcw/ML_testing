import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.linear_model import LinearRegression
import sys


def main():

    path = 'c:\\users\\dwright\\code\\'
    filename = path + 'scorebase.csv'
    readRows = 10000

    rnd = np.random.rand(readRows, 1) ** 3
    biny = np.round(rnd, 0) # binary y
    y = biny * np.random.lognormal(10, 2, (readRows, 1))
    y = y.astype('int')

    #print y
    train_data = pd.read_csv(filename, nrows=readRows)
    typeCols = zip(train_data.dtypes, train_data.columns)

    featureNames = ['protection_class', 'numberoffamilies', 'yearbuilt',
                    'NumberOfBathrooms', 'construction', 'SupplementalHeatingSource',
                    'Pool', 'WiringMaterial']
    fullFeatureNames = []
    #'DTC'
    featureNames = [i.lower() for i in featureNames]
    for item in typeCols:
        tp = item[0]
        colName = item[1]
        #print item
        if colName.lower() in featureNames:
            fullFeatureNames.append(colName)
            # I'd like the min, max, mean, mode and median of each item
            print tp, colName, train_data[colName][:10]

            if colName == 'Protection_Class':
                print 'fixing'
                train_data[train_data[colName] == '8B'] = '8'
                print 'fixed'
            if colName == 'NumberOfBathrooms':
                # for some reason this isn't working well... it is giving me an
                # empty value for the mean
                mn = train_data.loc[train_data[colName] != 'Unknown', colName].astype('float').mean()
                #mnAr = train_data[train_data[colName] != 'Unknown'].mean()
                print mn
                train_data.loc[train_data[colName] == 'Unknown', colName] = mn
            if colName.lower() == 'construction':
                train_data.loc[train_data[colName] == 'Framing, Wood', colName] = 'Frame'
                train_data.loc[train_data[colName] == 'Superior - Non '
                                                      'Combustible or Fire Resistive',
                                                      colName] = 'Superior'

            if colName == 'SupplementalHeatingSource':
                train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode()
            if colName == 'Pool':
                train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode()
            if colName == 'WiringMaterial':
                train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode()
                for forName in train_data[colName].value_counts().index:
                    # these are the index names of the rows
            # need to loop through them and if bleow a threshold, put into category of 'other'


    plt.clf()
    X_train = train_data.ix[:, fullFeatureNames]
    #clf = LinearRegression()
    clf = GradientBoostingRegressor()
    clf.fit(X_train, y)
    features = [0, 1, 2]

    fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=featureNames,
                                       n_jobs=3, grid_resolution=50)

    # run it!
    plt.show()


    # then the relative predictive power


if __name__ == '__main__':
    main()