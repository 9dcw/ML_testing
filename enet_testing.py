from sklearn import datasets
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import itertools

def main():
    seq = [[(i * .1, k * .1) for i in range(1, 3)] for k in range(1, 3)]
    seq = list(itertools.chain.from_iterable(seq))

    counter = 1
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    kfolds = KFold(X.shape[0], n_folds=4)
    for traini, testi in kfolds:
        alpha, l1 = seq[counter]
        print seq[counter]
        print alpha, l1
        enet = ElasticNet(alpha=alpha, l1_ratio=l1)
        y_pred = enet.fit(X[traini], y[traini]).predict(X[testi])
        score = r2_score(y[testi, y_pred])
        print score


if __name__ == '__main__':
    main()