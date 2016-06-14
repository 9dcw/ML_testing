import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression#, TheilSenRegressor, ElasticNet
from sklearn.cross_validation import train_test_split
import sys
from sklearn import preprocessing
import pylab as pl
import time
import statsmodels.api as sm
import xlsxwriter


def main():

    data = pd.DataFrame({'male':[1,1,0,0], 'female': [0,0,1,1],
                         'urban': [0,1,0,1], 'predict':[500, 800, 200, 400]})
    X = data[['male', 'female', 'urban']]
    Y = data['predict']
    wts = np.random.rand(X.shape[0])
    #wts = np.ones(X.shape[0])

    #Yhat, summary = model(X, Y, error='poisson', link='log', offset=wts)
    #sys.exit()
    run()

    return

def model(X, Y, error='poisson', link='log', wts=np.array([0]), offset=np.array([0]),
          names=np.array([0])):
    if wts.size == 1:
        wts = np.ones(X.shape[0])

    print error, link
    if link == 'log':
        lnk = sm.families.links.log
        lY = np.log(Y)
    else:
        lY = Y
        lnk = sm.families.links.identity

    if error == 'normal':
        fmly = sm.families.Gaussian(link=lnk)
        fmly2 = sm.families.Gaussian(link=lnk)
        lin = LinearRegression()
        lin.fit(X, lY, sample_weight=wts)
        #lin2 = LinearRegression()
        #lin2.fit(wX, wtY)

    elif error == 'poisson':
        fmly = sm.families.Poisson(link=lnk)
        fmly2 = sm.families.Poisson(link=lnk)

    else:
        print error, 'model not understood'
        sys.exit('model not understood')



    glm = sm.GLM(Y, X, family=fmly, offset=np.log(offset))
    ft = glm.fit()
    Yhat = ft.predict(X)

    return Yhat, ft.summary()

# right now I am reviewing the list of objectives and collecting my thoughts on how to model them
# I am also replicating the poisson model results from here

def run():
    t0 = time.time()
    path = 'c:\\users\\dwright\\code\\NBIC_glamming\\'
    filename = path + 'GLM_Variable_Inputs_Historical_distr.txt'
    readRows = None

    raw_train_data = pd.read_csv(filename, nrows=readRows)

    featureNames = ['pol_tenure', 'insrd_insuredage', 'prop_protectionclass', 'geo_dtc',
                    'Prop_SqFt', 'Prop_NumberOfFamilies', 'Prop_SqFt', 'Prop_DwellingAge',
                    'Geo_State', 'Pol_NumberOfLosses', 'Geo_County', 'Geo_Territory',
                    ]

    featureText = '''
        Exposure
        ,Geo_State
        ,Geo_ZipCode
        ,Geo_County
        ,Geo_Territory
        ,Geo_DTC
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
    incldFreq = {
    'Geo_State': True,
    'Insrd_InsuredAge': True,
    'Pol_Tenure': True,
    'Pol_AgentCategory': False,
    'Pol_AgentCategory_AccountCredit_TW': True,
    'Pol_CovA': True,
    'Pol_AOPDeductible': True,
    'Pol_NumberOfLosses': True,
    'Cred_CreditScore': True,
    'Prop_DwellingAge': True,
    'Prop_SqFt': True,
    'Prop_UsageType': True,
    'Prop_NumberOfBathrooms': True,
    'Prop_NumberOfFamilies': True,
    'Prop_NumberOfMajorHazards': True,
    'Prop_RoofLifeRemaining': True,
    'Prop_Pool': True,
    'Prop_SidingType': True,
    'Prop_SuppHeatSource': True,
    'geo_Mean_Travel_Work': True,
    'geo_Med_Val_Hous_Units': True,
    'geo_Persons_Per_Household': True,
    'geo_Pop_Per_Sq_Mile': True,
    'Claim_AccidentYear': True
    }

    removeList = ['Geo_ZipCode','Geo_County', 'Geo_Territory', 'Pol_Months_Since_Most_Recent_Loss']
    featureNames = [i.replace(' ', '').replace('\n','') for i in featureText.split(',') if
                    'Geo_' in i or 'Prop_' in i or 'Cred_' in i or 'Insurd_' in i or 'Pol_' in i]

    for i in featureNames:
        if i not in incldFreq:
            incldFreq[i] = False
    featureNames = [i for i in featureNames if i not in removeList]
    featureNames = [i[0] for i in zip(raw_train_data.columns, raw_train_data.dtypes)
                    if (i[0].lower() in featureNames or i[0] in featureNames)]

    two_way_features = ['Prop_NumberOfBathrooms', 'Prop_SqFt', 'Prop_CovA', 'Cred_CreditScore']

    print 'feature names', featureNames
    print 'raw data shape', raw_train_data.shape[0]
    outPath = path + '\\OutputCharts'
    train_data = raw_train_data.loc[:, featureNames]

    testCount1 = raw_train_data['Claim_Count'] > 0
    testCount2 = (raw_train_data['Claim_Loss_Incurred'] + raw_train_data['Claim_ALAE_Incurred']) > 0

    # need to make sure that the claims count fields always nonzero when the claims amount fields are
    assert testCount1.all() == testCount2.all()

    Xweights = raw_train_data.loc[:, 'Exposure']
    train_data = preprocess(train_data, bin=True)
    yList = ['Claim_Loss_Incurred', 'Claim_Count']#, 'Claim_ALAE_Incurred']
    raw_train_data[yList[0]] = raw_train_data[yList[0]] + raw_train_data['Claim_ALAE_Incurred']

    print 'calculating frequency regression'

    freqVector = raw_train_data[yList[1]]

    features = [i for i in incldFreq if incldFreq[i] == True]
    binned_features = []
    for ftr in train_data.columns:
        for tst in features:
            if tst in ftr:
                binned_features.append(ftr)
                break
        print train_data[ftr].dtype, list(train_data[ftr].unique()), ftr
    #print train_data[binned_features]
    X = np.asarray(train_data[binned_features]).astype('int64')
    X = train_data[binned_features].astype('int64')

    Y = np.asarray(freqVector).astype('int64')
    Y = freqVector

    Yhat, stats = model(X, Y, error='poisson', link='log', offset=Xweights)
    print Yhat

    # here I am building a graph of the predicted frequency against the actual frequency
    # the groupings will be predicted frequency bins
    outp = plotSingleHist('predicted_counts', np.ones(Yhat.shape), Yhat, Y,
                          outPath, Ytype='', Ycount=np.ones(Yhat.shape))
    print outp

    sys.exit()

    print 'calculating frequency regression'
    sevVector = raw_train_data[yList[0]]
    print 'calculating correlation'
    cmatrix = pd.DataFrame(train_data.corr())
    print 'writing correlation matrix'
    GenWriter = pd.ExcelWriter(outPath + '\\stats.xlsx', engine='xlsxwriter')
    cmatrix.to_excel(GenWriter, sheet_name='Correlation_Matrix')
    resultsMatrix.to_excel(GenWriter, sheet_name='regr_results')

    GenWriter.save()

    for Ytype in yList:
        X_tr, X_ts, Y_tr, Y_ts = train_test_split(train_data, raw_train_data.loc[:, Ytype], test_size=.3)
        print X_tr.shape
        lin.fit(X_tr, Y_tr)
        Yhat = lin.predict(X_ts)
        plt.scatter(Y_ts, Yhat)
        plt.show()
        plt.clf()
        plt.close()

        print 'params\n', lin.coef_()
        print 'score for ', Ytype, lin.score(X_ts, Y_ts)

    xlDict = {}
    # need to run through the columns and collect features that correspond to these
    # we already have a mapping of the column to an index

    print 'freq\n', freqVector.value_counts()
    for feat1 in two_way_features:
        feat1Vector, feat1Labels = vectorize(train_data, feat1)
        print 'feat1\n', feat1, feat1Vector.value_counts()
        for feat2 in two_way_features:
            feat2Vector, feat2Labels = vectorize(train_data, feat2)
            print 'feat2\n', feat2, feat2Vector.value_counts()
            #fig, ax = plt.subplots()

            #ax.scatter(x=feat1Vector, y=feat2Vector, s=freqVector)
            #ax.set_xticklabels(feat1Labels)
            #ax.set_yticklabels(feat2Labels)
            #plt.show()
            #plt.clf()
            #plt.close()

            #fig, ax = plt.subplots()
            #ax.scatter(x=feat1Vector, y=feat2Vector, s=secVector)
            #ax.set_xticklabels(feat1Labels)
            #ax.set_yticklabels(feat2Labels)

            #plt.show()
            #plt.clf()
            #plt.close()

    for Ytype in yList:
        bins = {}
        if Ytype == 'Claim_Loss_Incurred':
            Ycount = raw_train_data.loc[raw_train_data[Ytype] > 0, 'Claim_Count']
            X_train = train_data[raw_train_data[Ytype] > 0]
            Y = raw_train_data.loc[raw_train_data[Ytype] > 0, Ytype]
            Xweights = Xweights[raw_train_data[Ytype] > 0]
        elif Ytype == 'Claim_Count':
            Ycount = raw_train_data['Claim_Count']
            X_train = train_data
            Y = Ycount
            Xweights = raw_train_data['Exposure']
        else:
            print Ytype
            sys.exit("freq or sev?")
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

                #xlDict[Ytype + '_' + k + '_SR'] = plotSingleRegression(k, Xweights, train_data, Y, outPath, Ytype, Ycount)
                xlDict[Ytype + '_' + k + '_Hist'] = plotSingleHist(k, Xweights, X_train, Y, outPath, Ytype, Ycount)

        for bin in bins.keys():
            nm = Ytype + '_' + '_'.join(bins[bin][0].split('_')[:2]) + '_Hist'
            xlDict[nm] = plotMultiHistogram(bins[bin], Xweights, X_train, Y, outPath, Ytype, Ycount)

    FreqWriter = pd.ExcelWriter(outPath + '\\frequency.xlsx', engine='xlsxwriter')
    SevWriter = pd.ExcelWriter(outPath + '\\severity.xlsx', engine='xlsxwriter')
    xlDict[out].to_excel(SevWriter, sheet_name=sheetName)
    linkSheetF = FreqWriter.book.add_worksheet('index')
    linkSheetS = SevWriter.book.add_worksheet('index')
    Fcounter = 0
    Scounter = 0

    for out in xlDict:

        if 'Claim_Count' in out:
            Fcounter += 1
            if len(out) > 27:
                sheetName = str(Fcounter) + '_' + out[:27]
            else:
                sheetName = str(Fcounter) + '_' + out

            xlDict[out].to_excel(FreqWriter, sheet_name=sheetName)
            wbook = FreqWriter.book
            wksht = FreqWriter.sheets[sheetName]

            linkSheetF.write_url(row=Fcounter,col=1,url='internal:' + sheetName + '!a1', string=out)

        elif 'Claim_Loss_Incurred' in out:
            Scounter += 1
            if len(out) > 27:
                sheetName = str(Scounter) + '_' + out[:27]
            else:
                sheetName = str(Scounter) + '_' + out

            xlDict[out].to_excel(SevWriter, sheet_name=sheetName)
            wbook = SevWriter.book
            wksht = SevWriter.sheets[sheetName]

            linkSheetS.write_url(row=Scounter, col=1, url='internal:' + sheetName + '!a1', string=out)
        else:
            print Ytype
            sys.exit('what type is this?')

        r, c = xlDict[out].shape
        wksht.write(0, c+5, out)
        wksht.write_url(row=0, col=c+4, url='internal:index!a1', string='Back To Index')
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


def plotSingleRegression(k, Xweights, train_data,Y, outPath, Ytype, Ycount):
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


def plotMultiHistogram(k, Xweights, train_data, Y, outPath, Ytype, Ycount):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    w = 0.8 #width

    print k, Ytype, 'encoded field'
    # I want to plot the frequencies of the various components
    #subjData = train_data.loc[:, k] * Xweights
    xObsData = (Xweights*(train_data.loc[:, k]).T).T
    # remember that the train data is binary in each col
    # so this is returning the claims for those columns
    countData = (Ycount*(train_data.loc[:, k]).T).T

    if Ytype == 'Claim_Count':
        xObs = np.sum(xObsData)
        yObs = np.sum(countData) / xObs
    elif Ytype == 'Claim_Loss_Incurred':
        xObs = np.sum(countData)
        yObs = np.sum((Y*(train_data.loc[:, k]).T).T) / xObs
    else:
        print 'invalid Y type', Ytype
        sys.exit()
    xLabels = [str(i.split('_')[-1]) for i in k]
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
    plt.clf()
    plt.close()
    outAr = np.vstack((xObs, yObs)).T
    outFrame = pd.DataFrame(outAr, index=xLabels, columns=['Observations', 'values'])

    return outFrame


def plotSingleHist(k, Xweights, train_data, Y, outPath, Ytype, Ycount):
    print k
    if k == 'predicted_counts':
        trainerAr = train_data
        Ytype = k

    else:
        trainerAr = np.array(train_data.loc[:, k].astype('float'))

    try:
        # organize the data in to a 4 column array
        plotData = pd.DataFrame(np.vstack((trainerAr, Xweights, Y, Ycount)).T)
    except ValueError as e:
        print trainerAr.shape
        print Xweights.shape
        print Y.shape
        print Ycount.shape

        raise
    plotData.columns = [k, 'weights', 'target', 'counts']
    if np.unique(trainerAr).shape[0] > 10:
        # we'll need to bin the columns to build a histogram
        binCount = 10
        if k == 'predicted_counts':
            # I'll want to order by the Yhat (k) column
            plotData = plotData.sort_values(by=k, ascending=True)
            plotData.index = range(plotData.shape[0])
            #print plotData

            # now find the bins
            total = plotData['weights'].sum()

            bucketSize = total / binCount

            # then we need to allocate buckets
            # build cumulative sum
            refs = plotData['weights'].cumsum().astype('float')
            # now we are looking for integer switches
            refs = (refs / bucketSize).astype('int64')
            #print refs
            #initialize an offset array
            offRefs = pd.Series(np.zeros(trainerAr.shape[0]))
            #offset the array by one to detect changes
            offRefs[1:] = refs[:-1]
            # where the offset array is not equal that means we have
            # changed to a new bucket
            switches = refs != offRefs
            # detect the ones
            inds = switches.nonzero()[0]
            inds = inds.tolist()
            bins = plotData.ix[inds, 0].sort_values().tolist()

        else:
            bins = np.linspace(trainerAr.min(), trainerAr.max(), binCount)
        plotSmry = plotData.groupby(pd.cut(plotData[k], bins)).sum()
        # drop the first axis which is the sum of the binned values

        newVals = plotSmry[k]
        plotSmry.drop(k, axis=1, inplace=True)
        xLabels = [' to '.join(i.split(',')).replace('(','').replace(']','') for i in plotSmry.index]
    else:
        #there are few enough columns that we will just take them all in
        plotSmry = plotData.groupby(plotData.columns[0]).sum()
        xLabels = list(plotSmry.index)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    w = 0.8 #width

    # 0th is the weights because we drop the actual value
    if Ytype == 'Claim_Count' or 'predicted_counts':
        xObs = plotSmry['weights'].values
        yObs = plotSmry['target'].values

    else:
        xObs = plotSmry['counts'].values
        yObs = plotSmry['target'].values / xObs
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
    if k == 'predicted_counts':
        outAr = np.vstack((xObs, yObs, newVals)).T
        cols = ['Observations', 'values', 'expected']
    else:
        outAr = np.vstack((xObs, yObs)).T
        cols = ['Observations', 'values']
    outFrame = pd.DataFrame(outAr, index=xLabels, columns=cols)

    return outFrame

def preprocess(train_data, bin=False):

    typeCols = zip(train_data.dtypes, train_data.columns)
    count = train_data.shape[1]
    for item in typeCols:
        binSkip = True
        tp = item[0]
        colName = item[1]
        # we are binarizing all object fields
        if tp == 'object':
            binSkip = False
        # these are numeric fields we are still going to binarize

        if colName in ['Pol_Tenure', 'Pol_AOPDeductible', 'Pol_NumberOfLosses']:
            binSkip = False

        # leave these alone
        elif colName in ['Pol_CovA','Cred_CreditScore','Prop_DwellingAge', 'Prop_ProtectionClass',
                         'Prop_NumberOfBathrooms','Prop_NumberOfFamilies', 'Prop_SqFt',
                         'Prop_NumberOfMinorHazards', 'Prop_PanelType',
                         'Prop_RoofAge', 'Prop_NumberOfMajorHazards', 'Prop_RoofAge',
                         'Prop_RoofCoverType', 'Prop_SidingType', 'Prop_WiringMaterial']:

            pass #print colName, 'binskip', binSkip
        elif tp != 'object' or train_data.loc[:, colName].unique().shape[0] > 10:

            print train_data.loc[:, colName].unique()
            print item, train_data.loc[:, colName].unique().shape[0]
            sys.exit('new object')
        print colName
        #print train_data[colName].value_counts()
        if 'protectionclass' in colName.lower():
            print 'fixing', colName, tp
            if tp == 'object':
                train_data.loc[train_data.loc[:,colName] == '8B', colName] = '8'
            train_data.loc[:, colName] = train_data.loc[:, colName].astype('int64')
            train_data.loc[train_data.loc[:,colName] == 99, colName] = 10
            train_data.loc[train_data.loc[:,colName] == 1, colName] = 2

        elif colName == 'Pol_CovA':
            units = 100000
            train_data[colName] = train_data[colName].map(lambda x: int(round(x / units + .99999,0) * units))
            pd.to_numeric(train_data.loc[:, colName])

        elif colName == 'Cred_CreditScore':
            units = 25
            train_data[colName] = train_data[colName].fillna('Unknown')
            train_data.loc[train_data[colName] != 'Unknown', colName] = \
            train_data.loc[train_data[colName] != 'Unknown', colName].map(lambda x: str(int(round(int(x) / units + .99999,0) * units)))

        elif colName == 'Prop_RoofAge':
            units = 5
            train_data[colName] = train_data[colName].fillna('Unknown')
            train_data.loc[train_data[colName] != 'Unknown', colName] = \
            train_data.loc[train_data[colName] != 'Unknown', colName].map(lambda x: str(int(round(int(x) / units + .99999,0) * units)))

        elif colName == 'Prop_DwellingAge':
            units = 5

            if tp == 'object':
                train_data[colName] = train_data[colName].fillna('Unknown')
                train_data.loc[train_data[colName] != 'Unknown', colName] = \
                train_data.loc[train_data[colName] != 'Unknown', colName].map(lambda x: str(int(round(int(x) / units + .99999,0) * units)))

            else:
                train_data[colName] = train_data[colName].map(lambda x: str(int(round(int(x) / units + .99999,0) * units)))
                pd.to_numeric(train_data.loc[:, colName])

        elif colName == 'Prop_SqFt':
            units = 100
            if tp == 'object':
                train_data[colName] = train_data[colName].fillna('Unknown')
                train_data.loc[train_data[colName] != 'Unknown', colName] = \
                train_data.loc[train_data[colName] != 'Unknown', colName].map(lambda x: str(int(round(int(x) / units + .99999,0) * units)))

            else:
                train_data[colName] = train_data[colName].map(lambda x: str(int(round(int(x) / units + .99999,0) * units)))
                pd.to_numeric(train_data.loc[:, colName])
        else:
            if '%' in colName:
                train_data.loc[train_data.loc[:, colName].notnull(), colName] = \
                train_data.loc[train_data.loc[:, colName].notnull(), colName].map(lambda x: x.rstrip('%'))

            if tp == 'object':
                train_data.loc[train_data.loc[:, colName] == 'Unknown', colName] = np.nan
                train_data.loc[train_data.loc[:, colName] == 'No Historical Losses', colName] = 9999

            if (train_data[colName]).isnull().sum() > 0:
                try:
                    pd.to_numeric(train_data.loc[:, colName])
                    tp = 'int64'
                    print colName, 'converted to integer'
                except ValueError as e:
                    pass

        # we are activating the binning process and we
        # havn't excepted this variable from it via binSkip
        print count
        count -= 1
        if bin is True and binSkip == False:
            train_data[colName] = train_data[colName].fillna('Unknown')
            print 'encoding', colName, tp, 'binskip?', binSkip
            train_data, newCols = binarize(train_data, colName)
        else:
            # replacing unknowns with the modal value
            m = train_data[colName].value_counts().index[0]
            train_data[colName] = train_data[colName].fillna(m)

    return train_data

def binarize(train_data, colName):
    enc = preprocessing.LabelBinarizer()
    # fit the encoder
    try:
        enc.fit(train_data[colName])
    except ValueError as e:
        print train_data[colName]
        print train_data[colName].value_counts()
        print train_data[colName].isnull().sum()

        raise
        #sys.exit('fitting error')

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

def vectorize(train_data, feat1):
    featList = [i for i in train_data.columns if feat1 in i]
    featMatrix = train_data.loc[:, featList]
    c = 0
    for i in featList:
        # this will change the value in the matrix to the index value of the column
        featMatrix.loc[:, i] = featMatrix.loc[:, i] * c
        c += 1
    featVector = featMatrix.sum(axis=1)
    return featVector, featList

if __name__ == '__main__':
    main()