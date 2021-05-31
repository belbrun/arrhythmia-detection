import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from collections import Counter
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class_mapping = {'SB':0, 'SR':1, 'AFIB':2, 'ST':3, 'SVT':4, 'AF':5, 'SA':6,
                 'AT':7, 'AVNRT':8, 'AVRT':9, 'SAAWR':10}

feature_columns = ['PatientAge', 'Gender', 'VentricularRate', 'AtrialRate',
                   'QRSDuration', 'QTInterval', 'QTCorrected', 'RAxis', 'TAxis',
                   'QRSCount', 'QOnset', 'QOffset', 'TOffset']

class ShaoxingDataset(Dataset):
    def __init__(self, recordings, features, rhythms):
        self.recordings = recordings
        self.features = features
        self.rhythms = rhythms

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, index):
        if isinstance(index, list):
            X = [torch.Tensor(self.recordings[i]).to(device) for i in index]
            f = torch.Tensor(self.features[index]).to(device)
            y = torch.Tensor(self.rhythms[index]).to(device)
            return X, f, y

        return self.recordings[index], self.features[index], self.rhythms[index]

    def get_class_weights(self):
        counts = Counter(self.rhythms if self.rhythms.shape[1] == 1 else
                         np.argmax(self.rhythms))
        weights = np.empty(len(counts))
        for c in counts:
            weights[class_mapping[c]] = counts[c]
        return counts.most_common(1)[0][1]/weights

def load_diagnostics(path):
    return pd.read_excel(os.path.join(path, 'Diagnostics.xlsx'))


def load_recording(path):
    recording = pd.read_csv(path, sep=',',header=None)
    return recording.values[1:].astype('float32')

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

def map_gender(gender):
    return 0 if 'MALE' else 1

def get_splits(dataset_path, include=['train', 'valid', 'test']):
    splits = []
    for split in include:
        with open(os.path.join(dataset_path, split+'_rec.txt'), 'r+') as text_file:
            splits.append(text_file.read().splitlines())
    print(splits)
    return splits

def get_dataset(split, dataset_path, diagnostics, onehot=True, denoised=False):
    path = os.path.join(dataset_path, 'ECGDataDenoised' if denoised else 'ECGData')
    recordings = []
    for recording_path in split:
        recordings.append(load_recording(os.path.join(path,
                                                      recording_path +'.csv')))

    split = diagnostics.loc[diagnostics['FileName'].isin(split)]
    features = split[feature_columns]
    rhythms = split['Rhythm']
    features.loc[:, 'Gender'] = features['Gender'].apply(map_gender)
    return ShaoxingDataset(recordings, features.values,
                           map_annotations(rhythms.values, onehot))





def data_loaders(batch_size, shuffle=True, include=['train', 'valid', 'test']):
    dataset_path = os.path.join('dataset', '12lead')
    diagnostics = load_diagnostics(dataset_path)
    dss = []
    for split in get_splits(dataset_path, include):
        dss.append(get_dataset(split, dataset_path, diagnostics))

    samplers = [BatchSampler(RandomSampler(ds), batch_size=batch_size,
                           drop_last=False) for ds in dss]

    dls = [DataLoader(dss[i], sampler=sampler) for i, sampler in
           enumerate(samplers)]

    return dls, dss[0]



def get_length_frequencies(X):
    lengths = [x.shape[0] for x in X]
    length_frequencies = Counter(lengths)
    return length_frequencies.sorted(key=lambda pair: pair[0])

def save_log(log, path):
    with open(path, 'w+') as text_file:
        text_file.write('\n'.join(log))

def parse_log(path):
    with open(path, 'r') as text_file:
        log = text_file.readlines()
    train_loss, valid_loss, valid_acc = [], [], []
    for line in log:
        if line.startswith('Epoch') or line.startswith('Valid'):
            parts = line.split(':')
            if 'train' in parts[0]:
                train_loss.append(float(parts[1]))
            if 'validation' in parts[0]:
                valid_loss.append(float(parts[1]))
            if 'accuracy' in parts[0]:
                valid_acc.append(float(parts[1]))
    return train_loss, valid_loss, valid_acc

def split_dataset(path, train_size=0.6):
    diagnostics = load_diagnostics(path)
    recording_paths = diagnostics['FileName']
    rhythms = diagnostics['Rhythm']
    y = map_annotations(rhythms)
    x_train, x_validtest, y_train, y_validtest = train_test_split(recording_paths,
                                                        y,
                                                        train_size=train_size,
                                                        stratify=y)
    x_valid, x_test, y_valid, y_test = train_test_split(x_validtest,
                                                        y_validtest,
                                                        train_size=0.5,
                                                        stratify=y_validtest)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == '__main__':
    dl_test, ds_train = data_loaders(4, include=['test'])
    print('loaded')
    for i in dl_test[0]:
        print(i)
