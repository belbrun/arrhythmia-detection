import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np

files = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
         115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205,
         207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
         228, 230, 231, 232, 233, 234]
def load_example(path):
    signal = wfdb.rdsamp(path, sampto=None)
    annotation = wfdb.rdann(path, 'atr', sampto=None)
    return signal, annotation

def split_record(signal, sample):
    beats = []
    pos = sample[0]
    sample = sample[1:]
    for i in range(len(sample)-1):
        diff = (sample[i+1]-sample[i])//5
        beats.append(signal[pos: sample[i+1]-diff])
        pos = sample[i+1] - diff
    beats.append(signal[pos:])
    return beats

def get_mitbih():
    dataset_path = os.path.join('dataset', 'mit-bih'    )
    beats, annotations = [], []
    for i in files:
        s, a = load_example(os.path.join(dataset_path, str(i)))
        beats.extend(split_record(s[0], a.sample))
        annotations.extend(a.symbol[1:])
    return beats, annotations

def plot_record(path, sampto=None):
    record = wfdb.rdrecord(path, sampto=sampto)
    annotation = wfdb.rdann(path, 'atr', sampto=sampto)
    wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4), ecg_grids='all')

def plot(beat):
    t = np.arange(beat.shape[0])
    plt.plot(t, beat[:, 0])
    plt.show()


if __name__ == '__main__':
    X, y = get_mitbih()
    print(len(X), len(y))
