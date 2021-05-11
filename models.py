from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from data import filter_beats, pad_beats, get_mitbih, map_annotations

class Baseline:

    def __init__(self, kernel, n_features, use_preprocessing=True):
        self.svm = SVC(class_weight='balanced', kernel=kernel)
        self.n_features = n_features
        self.use_preprocessing = use_preprocessing

    def preprocess(self, X, y):
        X, y = filter_beats(X, y, self.n_features)
        return pad_beats(X, self.n_features), map_annotations(y)

    def fit(self, X, y):
        print('preparing')
        if self.use_preprocessing:
            X, y = self.preprocess(X, y)
        print(X.shape)
        print('fitting')
        self.svm.fit(X, y)


    def test(self, X, y):
        if self.use_preprocessing:
            X, y = self.preprocess(X, y)
        print('scoring')
        print(self.svm.score(X, y))

if __name__ == '__main__':
    X, y = get_mitbih()
    #X, y = X[:10000], y[:10000]
    print('split    ')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        stratify=y)
    #print(X_train)
    bl = Baseline('rbf', 500)
    bl.fit(X_train, y_train)
    bl.test(X_test, y_test)
