import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
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
    enc = preprocessing.LabelEncoder()
    for item in typeCols:
        tp = item[0]
        colName = item[1]
        #print item
        if colName.lower() in featureNames:
            fullFeatureNames.append(colName)
            # I'd like the min, max, mean, mode and median of each item
            print tp, colName#, train_data[colName][:10]

            if colName == 'Protection_Class':
                print 'fixing'
                train_data[train_data[colName] == '8B'] = '8'
                print 'fixed'
            if colName == 'NumberOfBathrooms':
                # for some reason this isn't working well... it is giving me an
                # empty value for the mean
                mn = train_data.loc[train_data[colName] != 'Unknown', colName].astype('float').mean()
                #mnAr = train_data[train_data[colName] != 'Unknown'].mean()
                #print mn
                train_data.loc[train_data[colName] == 'Unknown', colName] = mn
                train_data[colName] = train_data[colName].astype('float')
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
                c = 0
                for forName in train_data[colName].value_counts().index:
                    # these are the index names of the rows
                    ct = train_data[colName].value_counts()[forName]
                    if ct < train_data.shape[0] * .05:
                        print ct, forName
                        train_data.loc[train_data[colName] == forName, colName] = 'Other'

                    c += 1
                print train_data[colName].value_counts()
                # I'll need to look into this to see if some other groupings are appropriate
            retype = train_data[colName].dtype
            if retype == 'object':
                enc.fit(train_data[colName])

                train_enc = enc.transform(train_data[colName])
                train_data = train_data.drop(colName, 1)
                train_data = pd.concat((train_data, pd.DataFrame(train_enc)), 1)
                # I'm also going to need to name these encoding columns by the right name, rigtht?
                #train_data.rename(columns={'old':'new'}, inplace=True)

    X_train = train_data.ix[:, fullFeatureNames]
    clf = GradientBoostingRegressor()
    print X_train

    # it's telling me there's stuff in here that can't be a float
    # pool, wiring and heating source all got totally NaN'd
    clf.fit(X_train, y)
    features = [0, 1, 2]

    fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=featureNames,
                                       n_jobs=3, grid_resolution=50)

    # run it!
    plt.show()


    # then the relative predictive power

def oneHot():

    return

if __name__ == '__main__':
    main()