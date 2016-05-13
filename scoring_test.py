import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LinearRegression, TheilSenRegressor, ElasticNet
from sklearn.neighbors.kde import KernelDensity
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import sys
from sklearn import preprocessing
import pylab as pl
import time
import xlsxwriter

def main():
    t0 = time.time()
    path = 'c:\\users\\dwright\\code\\NBIC_glamming\\'
    filename = path + 'GLM_Variable_Inputs_Historical_distr.txt'
    readRows = 10000

    raw_train_data = pd.read_csv(filename, nrows=readRows)

    featureNames = ['pol_tenure', 'insrd_insuredage', 'prop_protectionclass', 'geo_dtc',
                    'Prop_SqFt', 'Prop_NumberOfFamilies', 'Prop_SqFt', 'Prop_DwellingAge',
                    'Geo_State', 'Pol_NumberOfLosses', 'Geo_County', 'Geo_Territory',
                    ]
    featureText = '''

        TransStart
        ,TransStop
        ,Exposure
        ,Geo_ZipCode
        ,Geo_County
        ,Geo_Territory
        ,Insrd_InsuredAge
        ,Pol_Tenure
         ,Pol_AgentCategory
        ,Pol_CovA
        ,Pol_AOPDeductible
        ,Pol_NumberOfLosses
        ,Pol_Months_Since_Most_Recent_Loss
        ,Pol_NumberOfPropLosses
        ,Pol_NumberOfLiabLosses
        ,Pol_Renovator%
        ,Pol_OrdOrLaw
        ,Pol_EDRC
        ,Pol_AccountCredit
        ,Pol_AtHomeCredit
        ,Pol_Umbrella
        ,Pol_LossFreeCredit%
        ,Cred_CreditScore
        ,Prop_DwellingAge
        ,Prop_Construction
        ,Prop_ProtectionClass
        ,Prop_SqFt
        ,Prop_UsageType
        ,Prop_FuelTankType
        ,Prop_Garage
        ,Prop_NumberOfBathrooms
        ,Prop_NumberOfFamilies
        ,Prop_NumberOfMinorHazards
        ,Prop_NumberOfMajorHazards
        ,Prop_PanelType
        ,Prop_RoofAge
        ,Prop_RoofCoverType
        ,Prop_RoofLifeRemaining
        ,Prop_PlumbingSupplyLines
        ,Prop_Pool
        ,Prop_SidingType
        ,Prop_SuppHeatSource
        ,Prop_UnderwriterAction
        ,Prop_WiringMaterial
        ,Claim_Count
        ,Claim_Loss_Incurred
        ,Claim_ALAE_Incurred
        ,Claim_AccidentYear
        ,Claim_PerilType
        '''

    featureNames = [i.replace(' ','').replace('\n','') for i in featureText.split(',') if
                    'Geo_' in i or 'Prop_' in i or 'Cred_' in i or 'Insurd_' in i or 'Pol_' in i]

    #featureNames = ['Geo_State', 'geo_dtc']

    featureNames = [i[0] for i in zip(raw_train_data.columns, raw_train_data.dtypes)
                    if (i[0].lower() in featureNames or i[0] in featureNames)]

    print 'feature names', featureNames
    print 'raw data shape', raw_train_data.shape[0]

    train_data = raw_train_data.loc[:, featureNames]

    Xweights = raw_train_data.loc[:, 'Exposure']

    train_data = preprocess(train_data, bin=True)
    yList = ['Claim_Count', 'Claim_Loss_Incurred']#, 'Claim_ALAE_Incurred']
    outPath = path + '\\OutputCharts'

    xlDict = {}
    #print train_data.columns

    for Ytype in yList:
        Y = raw_train_data.loc[:, Ytype]
        bins = {}
        for k in train_data.columns:
            if 'binned' in k:
                root = k.split('__')[0]
                if root not in bins.keys():
                    # first bianry in this list
                    bins[root] = [k]
                else:
                    # adding to binary group
                    oldBin = bins[root]
                    newBin = oldBin + [k]
                    bins[root] = newBin

            else: # not so run binary and this one

                xlDict[Ytype + '_' + k + '_SR'] = plotSingleRegression(k, Xweights, train_data, Y, outPath, Ytype)
                xlDict[Ytype + '_' + k + '_Hist'] = plotSingleHist(k, Xweights, train_data, Y, outPath, Ytype)

        for bin in bins.keys():
            xlDict[Ytype + '_' + '_'.join(bins[bin][0].split('_')[:2]) + '_Hist'] = plotMultiHistogram(bins[bin], Xweights, train_data, Y, outPath, Ytype)
    print len(xlDict)
    print xlDict.keys()
    FreqWriter = pd.ExcelWriter(outPath + '\\frequency.xlsx', engine='xlsxwriter')
    SevWriter = pd.ExcelWriter(outPath + '\\severity.xlsx', engine='xlsxwriter')
    c = 0
    for out in xlDict:
        c += 1
        if len(out) > 27:
            sheetName = str(c) + '_' + out[:27]
        else:
            sheetName = str(c) + '_' + out

        if 'Claim_Count' in Ytype:

            xlDict[out].to_excel(FreqWriter, sheet_name=sheetName)
            wbook = FreqWriter.book
            wksht = FreqWriter.sheets[sheetName]

        elif 'Claim_Loss_Incurred' in Ytype:

            xlDict[out].to_excel(SevWriter, sheet_name=sheetName)
            wbook = SevWriter.book
            wksht = SevWriter.sheets[sheetName]


        r, c = xlDict[out].shape
        wksht.write(r+5, 0, out)
        if '_Hist' in out:
            # certain chart treatment for histograms
            column_chart = wbook.add_chart({'type': 'column'})
            column_chart.add_series({
                'name': '={0}!B1'.format(sheetName),
                'categories': '={0}!A2:A{1}'.format(sheetName, r+1),
                'values': '={0}!B2:B{1}'.format(sheetName,r+1),
            })
            line_chart = wbook.add_chart({'type': 'line'})
            line_chart.add_series({
                'name': '={0}!C1'.format(sheetName),
                'categories': '={0}!A2:A{1}'.format(sheetName, r+1),
                'values': '={0}!C2:C{1}'.format(sheetName,r+1),
                'y2_axis': True,
            })
            column_chart.set_title({'name': out})
            column_chart.combine(line_chart)

            wksht.insert_chart('E2', column_chart)
        elif '_SR' in out:
            pass
            #different chart treatment for regressions

    SevWriter.save()
    FreqWriter.save()

    print (time.time() - t0) / 60
    return



def plotSingleRegression(k,Xweights, train_data,Y, outPath, Ytype):
    lin = LinearRegression()
    ax1 = pl.figure().add_subplot(111)
    print k, Ytype, 'numeric field'
    trainer = train_data.loc[:, k].astype('float').reshape(-1, 1)
    try:
        lin.fit(trainer, Y, sample_weight=Xweights)
    except ValueError as e:
        print trainer
        raise
    coef = lin.coef_[0]
    ax1.scatter(trainer, Y, color='black')
    prediction = lin.predict(trainer)
    ax1.plot(trainer, prediction, color='blue')
    ax1.set_ylabel(Ytype)
    #axes.annotate('coefficient:' + str(coef), xy=(float(trainer[-1]), float(prediction[-1]))
    #             ,xycoords='axes fraction',horizontalalignment='right', verticalalignment='right')
    t = k + '\ncoefficient: ' + str(coef)
    plt.title(t)
    plt.savefig(outPath + '\\' + k + '_' + Ytype + '_LinearRegression')
    plt.clf()
    plt.close()
    #print trainer.shape
    #print Y.shape
    #print prediction.shape
    out = pd.concat([pd.DataFrame(trainer), pd.DataFrame(Y), pd.DataFrame(prediction)], axis=1)

    return out


def plotMultiHistogram(k,Xweights, train_data,Y, outPath, Ytype):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    w = 0.8 #width

    print k, Ytype, 'encoded field'
    # I want to plot the frequencies of the various components
    #subjData = train_data.loc[:, k] * Xweights
    xObsData = (Xweights*(train_data.loc[:, k]).T).T
    # remember that the train data is binary in each col
    # so this is returning the claims for those columns
    claimsData = (Y*(train_data.loc[:, k]).T).T
    xObs = np.sum(xObsData)
    xLabels = [str(i.split('_')[-1]) for i in k]
    yObs = np.sum(claimsData) / xObs

    ax1.bar(range(xObs.shape[0]), xObs.values, width=w)

    name = '_'.join(k[0].split('_')[:2])
    tspots = [w/2 + i for i in range(xObs.shape[0])]
    ax2.plot(range(yObs.shape[0]), yObs.values, c='red')

    ax1.set_ylabel('observations')
    ax2.set_ylabel(Ytype)
    plt.subplots_adjust(bottom=0.25)

    ax1.set_xticks(tspots)
    ax1.set_xticklabels(xLabels, rotation=65)

    plt.subplots_adjust(bottom=0.25)
    plt.title(name)
    plt.savefig(outPath + '\\' + name + '_' + Ytype + '_Histogram')

    outAr = np.vstack((xObs, yObs)).T
    outFrame = pd.DataFrame(outAr, index=xLabels, columns=['Observations', 'values'])

    return outFrame

def plotSingleHist(k,Xweights, train_data,Y, outPath, Ytype):

    trainerAr = np.array(train_data.loc[:, k].astype('float'))

    plotData = pd.DataFrame(np.vstack((trainerAr, Xweights, Y)).T)
    plotData.columns = [k, 'weights', 'target']
    if np.unique(trainerAr).shape[0] > 10:
        # we'll need to bin the columns to build a histogram
        binCount = 10
        #print binCount
        bins = np.linspace(trainerAr.min(), trainerAr.max(), binCount)
        #print bins
        plotSmry = plotData.groupby(pd.cut(plotData.loc[:, plotData.columns[0]], bins)).sum()
        plotSmry.drop(plotSmry.columns[0], axis=1, inplace=True)
        xLabels = [' to '.join(i.split(',')).replace('(','').replace(']','') for i in plotSmry.index]
    else:
        #there are few enough columns that we will just take them all in
        plotSmry = plotData.groupby(plotData.columns[0]).sum()
        xLabels = list(plotSmry.index)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    w = 0.8 #width

    xObs = plotSmry.ix[:,0].values
    yObs = plotSmry.ix[:,1].values / xObs
    ax1.bar(range(plotSmry.shape[0]), xObs,width=w)
    tspots = [w/2 + i for i in range(plotSmry.shape[0])]
    ax1.set_xticks(tspots)
    ax1.set_xticklabels(xLabels, rotation=65)

    ax2.plot(range(plotSmry.shape[0]), yObs, c='red')
    ax1.set_ylabel('observations')
    ax2.set_ylabel(Ytype)
    plt.subplots_adjust(bottom=0.25)
    plt.title(s=k)
    #plt.show()
    plt.savefig(outPath + '\\' + k + '_' + Ytype + '_histogram')
    plt.clf()
    plt.close()
    #kde = KernelDensity(kernel='tophat').fit(trainer)
    #ht = kde.score_sampes(trainer)
    #print xObs
    #print yObs
    #print xLabels
    outAr = np.vstack((xObs,yObs)).T
    outFrame = pd.DataFrame(outAr, index=xLabels, columns=['Observations', 'values'])

    return outFrame

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
            #print train_data[colName].unique()
            #print colName, train_data.loc[:, colName].value_counts()

        #elif colName == 'NumberOfBathrooms':

        #    mn = train_data.loc[train_data[colName] != 'Unknown', colName].astype('float').mean()

         #   train_data.loc[train_data[colName] == 'Unknown', colName] = mn
         #   train_data.loc[:, colName] = train_data[colName].astype('float')
         #   pd.to_numeric(train_data.loc[:, colName])
        elif colName.lower() == 'construction':
            train_data.loc[train_data[colName] == 'Framing, Wood', colName] = 'Frame'
            train_data.loc[train_data[colName] == 'Superior - Non '
                                                  'Combustible or Fire Resistive',
                                                  colName] = 'Superior'

        elif colName == 'SupplementalHeatingSource':

            train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode().values[0]

        elif colName == 'Pool':
            m = train_data[colName].mode().values[0]

            train_data.fillna(m)
            train_data.loc[train_data[colName] == 'Unknown', colName] = m
            train_data.loc[train_data[colName] == '8', colName] = m

        #elif colName.lower() == 'insrd_insuredage':

        #    train_data.loc[train_data.loc[:, colName] == 'Unknown', colName] = np.nan
        #    m = train_data[colName].median()
        #    #print 'fixing', colName, m
        #    train_data.loc[:, colName] = train_data.loc[:, colName].fillna(m)
        #    pd.to_numeric(train_data.loc[:, colName])
        #    tp = 'int64'

        elif colName == 'Prop_NumberOfFamilies':
            pass
            #print train_data.loc[:, colName].describe()

        elif colName == 'WiringMaterial':
            train_data.loc[train_data[colName] == 'Unknown', colName] = train_data[colName].mode().values[0]
            c = 0
            # go through the list of materials
            for forName in train_data[colName].value_counts().index:
                # these are the index names of the rows
                ct = train_data[colName].value_counts()[forName]
                # if a material with a small number of instances, make it 'other'
                if ct < train_data.shape[0] * .05:
                    #print ct, forName
                    train_data.loc[train_data[colName] == forName, colName] = 'Other'

                c += 1
            # all looks like romex and BX.. is that right?
        else:
            if '%' in colName:
                train_data.loc[train_data.loc[:, colName].notnull(), colName] = \
                train_data.loc[train_data.loc[:, colName].notnull(), colName].map(lambda x: x.rstrip('%'))


            if tp == 'object':
                train_data.loc[train_data.loc[:, colName] == 'Unknown', colName] = np.nan
                train_data.loc[train_data.loc[:, colName] == 'No Historical Losses', colName] = 9999
                train_data.loc[train_data.loc[:, colName] == 'Unknown', colName] = np.nan

            if (train_data[colName]).isnull().sum() > 0:
                try:
                    pd.to_numeric(train_data.loc[:, colName])
                    tp = 'int64'
                except ValueError as e:
                    pass
                try:
                    m = train_data[colName].median()
                except TypeError as e:
                    m =  train_data.loc[:, colName].value_counts().index[0]
                train_data.loc[:, colName] = train_data.loc[:, colName].fillna(m)
        if bin is True and tp == 'object':
            train_data, newCols = binarize(train_data, colName)


    return train_data


def binarize(train_data, colName):
    enc = preprocessing.LabelBinarizer()
    print 'encoding', colName

    # fit the encoder
    try:
        enc.fit(train_data[colName])
    except ValueError as e:
        print train_data[colName]
        print e
        raise

    # build the encodings
    train_enc = enc.transform(train_data[colName])

    # zip up and dict the column names, which will be 'old: col_value'
    # col_value: colName_bin_#
    l = zip(range(len(enc.classes_)), enc.classes_)

    renameDict = {i[0]: colName + '__binned_' + str(i[1]) for i in l}

    # rename the new columns and build a dataframe
    train_enc = pd.DataFrame(train_enc).rename(columns=renameDict)
    # drop the old values
    #print train_data.columns
    train_data = train_data.drop(colName, 1)
    # concatenate in the new columns

    train_enc = pd.DataFrame(train_enc)
    train_data = pd.concat([train_data, train_enc], axis=1)
    print 'encoded', colName#, renameDict.values()
    return train_data, renameDict.values()

if __name__ == '__main__':
    main()