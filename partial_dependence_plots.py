

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing


cal_housing = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2,
                                                    random_state=1)
names = cal_housing.feature_names

clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                learning_rate=.1, loss='huber',
                                random_state=1)

print 'training'
clf.fit(X_train, y_test)
print 'trained'

features = [0, 5, 1, 2, (5, 1)]
fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=names,
                                   n_jobs=3, grid_resolution=50)

fig.suptitle('Partial dependence of house value on nonlocation features\n' + 'for the California housing dataset')
plt.subplots_adjust(top=0.9)

sys.exit()
print '_' * 80
print 'custom 3d plot via "partial_dependence'

#def plot_partial_dependence(clf, X_train, features, feature_names, n_jobs)

fig = plt.figure()

target_feature = (1,5)
pdp, (x_axis, y_axis) = partial_dependence(clf, target_feature, X=X_train,
                                           grid_resolution=50)

XX, YY = np.meshgrid(x_axis, y_axis)
Z = pdp.T.reshape(XX.shape).T

ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)

ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('partial dependence')
# pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('partial dependence of house value on median age and avg occupancy')

plt.subplots_adjust(top=0.9)

plt.show()