print 'importing'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print 'imported'

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

features = ['protection_class', 'numberoffamilies', 'yearbuilt']


for item in typeCols:
    tp = item[0]
    colName = item[1]
    #print colName
    if colName.lower() in features:
        # I'd like the min, max, mean, mode and median of each item
        print tp, colName, train_data[colName][:10]
        print train_data[colName].describe()
        #print train_data[colName].unique()
        if colName == 'Protection_Class':
            runData = train_data[colName]
            runData[runData == '8B'] = '8'
            print runData.shape
            # run it!

            plt.scatter(np.array(runData).astype('int'), biny)

        else:
            plt.scatter(train_data[colName], biny)
        plt.show()


# want to put together a scatter of these on a multi-dimensional basis
# need to define the feature sets, pick a model to train, pass the names

features = [0, 5, 1, 2, (5, 1)]
fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=names,
                                   n_jobs=3, grid_resolution=50)


# how about plot of partial dependnece!

# then the relative predictive power