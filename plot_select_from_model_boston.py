import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected.
# With Lasso, the higher the alpha parameter, the fewer features selected.

# see this http://statweb.stanford.edu/~tibs/lasso/simple.html

boston= load_boston()
X, y = boston['data'], boston['target']

# we use the base estimator LassocCV since the L1 norm promotes sparsity of features
# importantly, lassocv is run with the least angle regression, as discussed in the link above.

clf = LassoCV()

# Set a minimum threshold of 0.25
# this is a 'maxing out' of the sum of all coefficients
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)

n_features = sfm.transform(X).shape[1]

# reset the threshold until the number of features equals two.
# Note that the attribute can be set directly instead of repeatedley
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the seelcted two features from X.
plt.title('features selected from boston using the SelectFromModel with'
          'threshold of %0.3f.' % sfm.threshold)

feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Value of Feature number 1")
plt.ylabel("Value of Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()

