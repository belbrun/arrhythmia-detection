import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np

files = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
         115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205,
         207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
         228, 230, 231, 232, 233, 234]

class_mapping = {'R':0, 'L':1, 'Q':2, '|':3, 'N':4, 'A':5, 'J':6, 'f':7,
                 'x':8, 'S':9, 'j':10, 'e':11, 'E':12, 'a':13, '/':14, 'F':15,
                 'V':16, }

def load_example(path):
    signal = wfdb.rdsamp(path, sampto=None)
    annotation = wfdb.rdann(path, 'atr', sampto=None)
    return signal, annotation

def filter_annotations(sample, symbol):
    filtered_symbol, filtered_sample = [], []
    for i in range(len(symbol)):
        if symbol[i] in class_mapping.keys():
            filtered_sample.append(sample[i])
            filtered_symbol.append(symbol[i])
    return np.array(filtered_sample), filtered_symbol

def split_record(signal, sample, max_len=1750):
    beats = []
    pos = sample[0]*4//5
    for i in range(len(sample)-1):
        diff = (sample[i+1]-sample[i])//5
        beats.append(signal[pos: sample[i+1]-diff])
        pos = sample[i+1] - diff
    beats.append(signal[pos:])
    return beats

def filter_beats(beats, annotations, max_len):
    to_filter = [i for i, x in enumerate(beats) if x.shape[0] > max_len]
    to_filter.reverse()
    for i in to_filter:
        del beats[i]
        del annotations[i]
    return beats, annotations

def map_annotations(annotations, onehot=False):
    if onehot:
        Y = np.zeros((len(annotations), len(class_mapping.keys())))
    else:
        Y = np.empty(len(annotations))
    for i, annotation in enumerate(annotations):
        if onehot:
            Y[i, class_mapping[annotation]] = 1
        else:
            Y[i] = class_mapping[annotation]
    return Y

def pad_beats(beats):
    longest = max([x.shape[0] for x in beats])
    X = np.empty((len(beats), longest), dtype='float32')
    for i, beat in enumerate(beats):
        X[i] = np.pad(beat[:, 0], (0, longest-beat.shape[0]), 'constant',
                      constant_values=0)
    return X


def get_mitbih():
    dataset_path = os.path.join('dataset', 'mit-bih'    )
    beats, annotations = [], []
    for i in files:
        s, a = load_example(os.path.join(dataset_path, str(i)))
        sample, symbol = filter_annotations(a.sample,
                                            a.symbol)
        beats.extend(split_record(s[0], sample))
        annotations.extend(symbol)
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

"""
if __name__ == '__main__':
    print('get')
    X, y = get_mitbih()
    print('filter')
    X, y = filter_beats(X, y, 1750)
    print('pad')
    X = pad_beats(X)
    print(X)
"""
