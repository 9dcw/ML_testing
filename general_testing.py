print 'loading libraries'
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn import preprocessing


import sys
from sklearn.grid_search import GridSearchCV


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
pca = PCA(n_components=2)
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
                  svm__C=[0.1, 1, 10])from sklearn.grid_search import GridSearchCV

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print grid_search.best_estimator_
sys.exit()

lr = linear_model.LinearRegression()

print X.shape, y.shape
predicted = cross_val_predict(lr, X, y, cv=10)
print predicted.shape
fig, ax = plt.subplots()

ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()