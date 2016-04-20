import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import sys
from sklearn import preprocessing

def main():

    path = 'c:\\users\\dwright\\code\\NBIC_glamming\\'
    filename = path + 'GLM_Variable_Inputs_Historical_distr.txt'
    readRows = 1000

    raw_train_data = pd.read_csv(filename, nrows=readRows)

    featureNames = ['pol_tenure', 'insrd_insuredage', 'prop_protectionclass', 'geo_dtc',
                    'Prop_SqFt', 'Prop_NumberOfFamilies', 'Prop_SqFt', 'Prop_DwellingAge']

    featureNames = [i[0] for i in zip(raw_train_data.columns, raw_train_data.dtypes)
                    if (i[0].lower() in featureNames or i[0] in featureNames)]# and i[1] != 'O']

    print 'feature names', featureNames
    print 'raw data shape', raw_train_data.shape[0]

    train_data = raw_train_data.loc[:, featureNames]

    Xweights = raw_train_data.loc[:, 'Exposure']
    claimsCounts = raw_train_data.loc[:, 'Claim_Count']
    claimsAmounts = raw_train_data.loc[:, 'Claim_Loss_Incurred']
    ALAEAmounts = raw_train_data.loc[:, 'Claim_ALAE_Incurred']

    kf = KFold(train_data.shape[0], n_folds=2)
    train_data = preprocess(train_data, bin=True)

    bins = {}
    outPath = path + '\\OutputCharts'
    Y = claimsAmounts

    #clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
    #                                 max_depth=1, random_state=0)

    clf = GradientBoostingRegressor()
    clf.fit(train_data, Y, sample_weight=Xweights)


    fig, axs = plot_partial_dependence(clf, train_data, features=range(train_data.shape[1]), feature_names=featureNames,
                                       n_jobs=3, grid_resolution=50)
    plt.show()
    sys.exit()


    for k in train_data.columns:
        if 'binned' in k:
            root = k.split('_')[0]
            if root not in bins.keys():
                # first bianry in this list
                bins[root] = [k]
            else:
                # adding to binary group
                bins[root] = bins[root].append(k)
        else: # not so run binary and this one
            plotLinearScats(k, train_data, Y, Xweights, outPath)

    for bin in bins.keys():
        # run the bins
        plotLinearScats(bins[bin], train_data, Y, Xweights, outPath)


    return

def plotLinearScats(k, train_data, Y, Xweights, outPath):
    lin = LinearRegression()
    if len(list(k)[0]) == 1: # this is not a list
        trainer = train_data.loc[:, k].reshape(-1,1)
        lin.fit(trainer, Y, sample_weight=Xweights)
        plt.scatter(trainer, Y, color='black')
        plt.plot(trainer, lin.predict(trainer), color='blue')
    else:
        pass
    print lin.coef_
    t = k + '\ncoefficient: ' + str(lin.coef_[0])
    plt.title(t)
    plt.savefig(outPath + '\\' + k + '_LinearRegression')

    return


def backup():

    for ftr in featureNames:
        print ftr, train_data.loc[:, ftr].describe()

    #clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
    #                                 max_depth=1, random_state=0)

    clf = RandomForestRegressor()
    clf.fit(train_data, claimsCounts, sample_weight=Xweights)

    # I want to show partial dependence in two ways: linear relationship and partial dependence

    # partial dependence:
    # For each value of the target features in the grid the partial dependence function need to marginalize
    # the predictions of a tree over all possible values of the complement features.
    # In decision trees this function can be evaluated efficiently without reference to the training data.
    # For each grid point a weighted tree traversal is performed:
    # if a split node involves a target feature, the corresponding left or right branch is followed,
    # otherwise both branches are followed, each branch is weighted by the
    # fraction of training samples that entered that branch.
    # Finally, the partial dependence is given by a weighted average of all visited leaves.
    # For tree ensembles the results of each individual tree are again averaged.

    print featureNames
    fig, axs = plot_partial_dependence(clf, train_data, features=range(train_data.shape[1]), feature_names=featureNames,
                                       n_jobs=3, grid_resolution=50)

    plt.title('Dependence of Variable on Claims Frequency')
    plt.savefig(path + 'partial_dependence.png')

    plt.clf()

    pdp, (x_axis, y_axis) = partial_dependence(clf, (0,1), X=train_data)

    XX, YY = np.meshgrid(x_axis, y_axis)

    Z = pdp.T.reshape(XX.shape)
    print XX.shape, YY.shape, Z.shape
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
    plt.colorbar(surf)
    plt.subplots_adjust(top=0.9)
    plt.show()

    pivot_train_data = pd.concat([train_data, raw_train_data.loc[:, ['Claim_Count', 'Exposure']]], axis=1)
    print pivot_train_data

    print pd.pivot_table(pivot_train_data, values=['Exposure', 'Claim_Count'],
                         index='Prop_ProtectionClass', aggfunc=np.sum)


    sys.exit()


    # let's fit a random forest!
    depths = [i * 2 for i in range(1,5)]
    r2s = []
    importances = []
    for depth in depths:
        print 'fitting depth', depth
        randforest = RandomForestRegressor(max_depth=depth)
        randforest.fit(X_train, y_train)
        imp = randforest.feature_importances_
        print imp
        plt.plot(range(X_train.shape[1]), imp)
        #plt.set_xlabel(X_train.columns)
        plt.show()
        sys.exit()
        importances.append(imp)

        y_predict = randforest.predict(X_test)
        #plt.scatter(y_predict, y_test)
        r2s.append(r2_score(y_test, y_predict))

    plt.scatter(r2s, depths)
    plt.show()
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


def preprocess(train_data, bin=False):

    typeCols = zip(train_data.dtypes, train_data.columns)

    for item in typeCols:
        tp = item[0]
        colName = item[1]

        # I'd like the min, max, mean, mode and median of each item
        print tp, colName #, train_data[colName][:10]

        if 'protectionclass' in colName.lower():
            print 'fixing', colName, tp
            if tp == 'object':
                train_data.loc[train_data.loc[:,colName] == '8B', colName] = '8'
            train_data.loc[:, colName] = train_data.loc[:, colName].astype('int64')
            train_data.loc[train_data.loc[:,colName] == 99, colName] = 10
            train_data.loc[train_data.loc[:,colName] == 1, colName] = 2
            print train_data[colName].unique()
            print colName, train_data.loc[:, colName].value_counts()

        if colName == 'NumberOfBathrooms':

            mn = train_data.loc[train_data[colName] != 'Unknown', colName].astype('float').mean()

            train_data.loc[train_data[colName] == 'Unknown', colName] = mn
            train_data.loc[:, colName] = train_data[colName].astype('float')
            pd.to_numeric(train_data.loc[:, colName])
        if colName.lower() == 'construction':
            train_data.loc[train_data[colName] == 'Framing, Wood', colName] = 'Frame'
            train_data.loc[train_data[colName] == 'Superior - Non '
                                                  'Combustible or Fire Resistive',
                                                  colName] = 'Superior'

        if colName == 'SupplementalHeatingSource':

            train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode().values[0]

        if colName == 'Pool':
            m = train_data[colName].mode().values[0]

            train_data.fillna(m)
            train_data.loc[train_data[colName] == 'Unknown', colName] = m
            train_data.loc[train_data[colName] == '8', colName] = m

        if colName.lower() == 'insrd_insuredage':

            #sys.exit()
            train_data.loc[train_data.loc[:, colName] == 'Unknown', colName] = np.nan
            m = train_data[colName].median()
            #print 'fixing', colName, m
            train_data.loc[:, colName].fillna(m)
            pd.to_numeric(train_data.loc[:, colName])
            tp = 'int64'

        if colName == 'Prop_NumberOfFamilies':

            print train_data.loc[:, colName].describe()
            # it's failing here...

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
            # all looks like romex and BX.. is that right?

        if bin is True and tp == 'object':
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
    # col_value: colName_bin_#
    l = zip(range(1, len(enc.classes_)+1), enc.classes_)
    renameDict = {i[0]: colName + '_binned_' + str(i[1]) for i in l}

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