print 'loading libraries'
from sklearn import datasets, decomposition
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn import svm
#from sklearn.cross_validation import cross_val_predict
#from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence



import sys
from sklearn.grid_search import GridSearchCV



def main():
    print 'loading data'
    boston = datasets.load_boston()
    #iris = datasets.load_iris()
    print 'data loaded'
    X, y = boston.data, boston.target
    #X, y = iris.data, iris.target
    print 'X shape:', X.shape
    print 'y shape:', y.shape

    scaler = StandardScaler()

    # I need to fit and transform the data with the scaler.. how do I put
    # this into pipeline?


    # initialize PCA to pick 5 components
    #pca = decomposition.PCA(n_components=4)

    scaledX = scaler.fit_transform(X)
    #kf = cross_validation.KFold(scaledX, n_folds=3, shuffle=True)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(scaledX, y)

    # then I will plot partial dependence to see how the features work
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=.1, loss='huber',
                                    random_state=1)

    print 'training', X_train.shape, y_train.shape
    clf.fit(X_train, y_train)
    print 'trained'

    features = [0, 5, 1, 2, (5, 1)]
    fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=None,
                                       n_jobs=3, grid_resolution=50)

    fig.suptitle('Partial dependence of house value on nonlocation features\n' + 'for the California housing dataset')
    plt.subplots_adjust(top=0.9)

    plt.show()

    # then I will PCA and plot partial dependence there
    # then lasso PCA

    # then select the parameters
    # plots are nice but shouldn't I be selecting based on multi-dimensional data?

    # then generate a list of multi-parameter algorithms
    # then do a parameter search with gradient boost and a few other multi-parameter algorithms


    return


def old():
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained variance')
    plt.show()

    # initialize selectKBest to pick the best of the ones that occur naturally
    # the feature union does not check whether there is overlap between the estimators
    # so we need to seriously watch out for this...
    selection = SelectKBest(k=1)

    # build a dict with these for pipeline purposes
    combined_features = FeatureUnion([('pca', pca), ('univ_select', selection)])

    # use the combined features to transform the dataset
    X_features = combined_features.fit(X, y).transform(X)

    # initialize the svm
    svm = SVR(kernel="linear")

    # I think I put the scaler into the first set of the pipeline...

    pipeline = Pipeline([('scaler', scaler), ('features', combined_features), ('svm', svm)])

    param_grid = dict(features__pca__n_components=[2, 5, 10],
                      features__univ_select__k=[1, 2],
                      svm__C=[0.1, 1, 10])

    #scoring: precision, accuracy, recall,

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    grid_search.fit(X, y)
    print grid_search.best_estimator_

    scores = grid_search.grid_scores_


    fig, ax = plt.subplots()

    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()



if __name__ == '__main__':
    main()