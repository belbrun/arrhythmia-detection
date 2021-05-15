import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pywt
import drawer
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split


files = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
         115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205,
         207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
         228, 230, 231, 232, 233, 234]

class_mapping = {'R':0, 'L':1, 'Q':2, '|':3, 'N':4, 'A':5, 'J':6, 'f':7,
                 'x':8, 'S':9, 'j':10, 'e':11, 'E':12, 'a':13, '/':14, 'F':15,
                 'V':16}

class MITBIHDataset(Dataset):
    def __init__(self, beats, annotations):
        self.beats = beats
        self.annotations = annotations

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, index):
        return self.beats[index], self.annotations[index]


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

def collate(batch):
    print(len(batch))
    X, y = batch
    return Tensor(pad_beats(X)), map_annotations(y, onehot=True)

def pad_beats(beats, pad_size=None):
    longest = max([x.shape[0] for x in beats])
    pad_size = longest if pad_size is None or pad_size < longest else pad_size
    X = np.empty((len(beats), pad_size), dtype='float32')
    for i, beat in enumerate(beats):
        X[i] = np.pad(beat[:, 0], (0, pad_size-beat.shape[0]), 'constant',
                      constant_values=0)
    return X

def denoise(X):
    print('denoising')
    X_denoised = []
    for x in X:
        thd = np.empty(x.shape)
        thd[0] = pywt.threshold(x[0], np.median(x[0]), 'hard')
        thd[1] = pywt.threshold(x[1], np.median(x[1]), 'hard')
        X_denoised.append(thd)
    return X_denoised


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

def data_loaders(batch_size, shuffle=True, ratios=[0.6, 0.2, 0.2]):
    X, y = get_mitbih()
    X_train, X_testvalid, y_train, y_testvalid = train_test_split(X, y,
                                                                  train_size=\
                                                                  ratios[0],
                                                                  shuffle=True,
                                                                  stratify=y)
    print(len(X_testvalid), len(y_testvalid))
    X_valid, X_test, y_valid, y_test = train_test_split(X_testvalid,
                                                        y_testvalid,
                                                        train_size=ratios[1]/\
                                                        (ratios[1]+ratios[2]))

    ds_train = MITBIHDataset(X_train, y_train)
    ds_valid = MITBIHDataset(X_valid, y_valid)
    ds_test = MITBIHDataset(X_test, y_test)

    dl_train = DataLoader(ds_train, batch_size, shuffle, collate_fn=collate)
    dl_valid = DataLoader(ds_valid, batch_size, shuffle, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size, shuffle, collate_fn=collate)

    return dl_train, dl_valid, dl_test



def get_class_weights(y):
    pass




def get_length_frequencies(X):
    lengths = [x.shape[0] for x in X]
    length_frequencies = Counter(lengths)
    return length_frequencies.sorted(key=lambda pair: pair[0])



if __name__ == '__main__':
    print('get')
    X, y = get_mitbih()
    drawer.plot_beat(denoise(X[:100])[50])
    drawer.plot_beat(X[50])
    c = Counter(y)
    print(c)
