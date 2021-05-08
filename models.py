from sklearn.svm import SVC
from data import filter_beats, pad_beats, get_mitbih, map_annotations

class Baseline:

    def __init__(self):
        self.svm = SVC(class_weight='balanced')

    def prepare(self, X, y):
        X, y = filter_beats(X, y, 1750)
        return pad_beats(X), map_annotations(y)

    def fit(self, X, y):
        X, y = self.prepare(X, y)
        self.svm.fit(X, y)
        print(self.svm.score(X, y))

if __name__ == '__main__':
    X, y = get_mitbih()
    bl = Baseline()
    bl.prepare(X, y)
    bl.fit(X, y)
