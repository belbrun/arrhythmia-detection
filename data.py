import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from collections import Counter
from sklearn.model_selection import train_test_split

# use GPU for training if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# names of ECG recording files
files = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
         115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205,
         207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
         228, 230, 231, 232, 233, 234]

# class annotations
class_mapping = {'R':0, 'L':1, 'Q':2, '|':3, 'N':4, 'A':5, 'J':6, 'f':7,
                 'x':8, 'S':9, 'j':10, 'e':11, 'E':12, 'a':13, '/':14, 'F':15,
                 'V':16}

random_state = 42 # set to 42 to reproduce split used in the project

class MITBIHDataset(Dataset):
    """
        Model the MIT-BIH arrythmia dataset.

        Atributes
        ----------
        beats: list[numpy.array]
            ECG recorings of one hearthbeat
        annotations: list[str]
            which represent heartbeats class
    """

    def __init__(self, beats, annotations):
        self.beats = beats
        self.annotations = annotations

    def __len__(self):
        """
        Return number of examples in the dataset
        -------
        length: int
            Number of examples in the dataset.
        """
        return len(self.beats)

    def __getitem__(self, index):
        if isinstance(index, list):
            X = [torch.Tensor(self.beats[i]).to(device) for i in index]
            y = [self.annotations[i] for i in index]
            return X, y

        return self.beats[index], self.annotations[index]

    def get_class_weights(self):
        """
            Calculate class weights.

            Class weight is calculated as cardinality of the class scaled with
            the cardinality of the class with most examples.
        """
        counts = Counter(self.annotations)
        weights = np.empty(len(counts))
        for c in counts:
            weights[class_mapping[c]] = counts[c]
        return counts.most_common(1)[0][1]/weights



def load_example(path):
    """
        Load a single ECG recording from the MIT-BIH database.

        Parameters
        ----------
        Retuns a tuple containing the signal and annotations of the recording.
    """
    signal = wfdb.rdsamp(path, sampto=None)
    annotation = wfdb.rdann(path, 'atr', sampto=None)
    return signal, annotation

def filter_annotations(sample, symbol):
    """
        Filter annotations and annotation positions to keep only the ones that
        refer to the heathbeat.

        Parameters
        ----------
        sample: list[int]
            Positions of annotations in the recording.
        symbol: list[str]
            Anotations.

        Returns
        -------
        sample: numpy.ndarray
            Filtered positions of annotations in the recording.
        symbol: list[str]
            Filtered annotations.
    """
    filtered_symbol, filtered_sample = [], []
    for i in range(len(symbol)):
        if symbol[i] in class_mapping.keys():
            filtered_sample.append(sample[i])
            filtered_symbol.append(symbol[i])
    return np.array(filtered_sample), filtered_symbol

def split_record(signal, sample):
    """
        Split recording into heartbeats according to the positions of
        annotations.
        Uses an algorithm to determine the start and the end coordinate of the
        hearthbeat from the annotations coordinate, which coresponds to the
        coordinate of the heatbeats R-wave peak.

        Parameters
        ----------
        sample:  numpy.ndarray
            Positions of annotations in the recording.
        symbol: list[str]
            Anotations.
    """
    beats = []
    pos = sample[0]*4//5
    for i in range(len(sample)-1):
        diff = (sample[i+1]-sample[i])//5
        beats.append(signal[pos: sample[i+1]-diff])
        pos = sample[i+1] - diff
    beats.append(signal[pos:])
    return beats

def get_split_positions(signal, sample):
    positions = [sample[0]*4//5]
    for i in range(len(sample)-1):
        diff = (sample[i+1]-sample[i])//5
        positions.append(sample[i+1] - diff)
    return positions

def filter_beats(beats, annotations, max_len):
    """
        Filter beats longer then max_length.

        Parameters
        ----------
        beats:  list[numpy.ndarray]
            Heartbeat recordings.
        symbol: list[str]
            Heartbeat annotations.

        Returns
        -------
        beats:  list[numpy.ndarray]
            Filtered heartbeat recordings.
        symbol: list[str]
            Filtered heartbeat annotations.

    """
    to_filter = [i for i, x in enumerate(beats) if x.shape[0] > max_len]
    to_filter.reverse()
    for i in to_filter:
        del beats[i]
        del annotations[i]
    return beats, annotations

def map_annotations(annotations, onehot=False):
    """
        Map annotation symbols to numerical values or to one-hot encoded
        vectors.

        Parameters
        ----------
        annotations: list[str]
            Annotation symbols.
        onehot: bool
            Flag to determine if the return should be a one hot encoded vector.

        Returns
        -------
        annotations: np.ndarray
            Array containing numerical values or one-hot encoded vectors of
            annotations.
    """
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
    """
        Pad a mini batch of heartbeat recordings and maps annotations.

        Parameters
        ----------
        batch: tuple(numpy.ndarray, list[str])
            Batch of examples.

        Returns
        -------
        batch: tuple(torch.Tensor, torch.Tensor)
            Batch of examples.
    """
    X, y = batch[0]
    return pad_sequence(X, batch_first=False), \
            torch.LongTensor(map_annotations(y)).to(device)

def pad_beats(beats, pad_size=None):
    """
        Pad beats to have the same size in dimension 0.

        Parameters
        ----------
        beats: list[numpy.ndarray]
            Heartbeat recordings.
        pad_size: int
            Value to pad to. If not specified or if its lower then the length of
            the longest element, it is set to the length of the longest element.

        Returns
        -------
        beats: numpyp.ndarray
            Padded heartbeat recordings.


    """
    longest = max([x.shape[0] for x in beats])
    pad_size = longest if pad_size is None or pad_size < longest else pad_size
    X = np.empty((len(beats), pad_size), dtype='float')
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
    """
        Load MIT-BIH dataset recordings and annotations as separate heartbeats.

        Returns
        -------
        beats: list[numpy.ndarray]
            Heartbeat recordings.
        annotations: list[str]
            Hearbeat annotations.
    """
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
    """
        Generate batch iterators over train, validation and test splits of the
        dataset.
        Use train_test_split to split the original dataset in given ratios.
        Generate BatchSamplers over MITBIHDataset objects, then uses it to
        generate DataLoader objects.

        Parameters
        ----------
        batch_size: int
            Batch size.
        shuffle: bool
            Flag to determine if the data is shuffled before spliting.
        ratios: list[float]
            Ratios to split the datasets in.

        Returns
        -------
        data_loaders: list[DataLoader]
            List of train, validation and test data loaders, respectfully.
        datasets:
            List of train, validation and test datasets, respectfully.
    """
    X, y = get_mitbih()
    X_train, X_testvalid, y_train, y_testvalid = train_test_split(X, y,
                                                                  train_size=\
                                                                  ratios[0],
                                                                  shuffle=True,
                                                                  stratify=y,
                                                                  random_state=\
                                                                  random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_testvalid,
                                                        y_testvalid,
                                                        train_size=ratios[1]/\
                                                        (ratios[1]+ratios[2]),
                                                        random_state=\
                                                        random_state)

    dss = [MITBIHDataset(X, y) for X, y in [(X_train, y_train),
                                            (X_valid, y_valid),
                                            (X_test, y_test)]]

    samplers = [BatchSampler(RandomSampler(ds), batch_size=batch_size,
                           drop_last=False) for ds in dss]

    dls = [DataLoader(dss[i], sampler=sampler, collate_fn=collate)
           for i, sampler in enumerate(samplers)]

    return dls, dss


def get_length_frequencies(X):
    lengths = [x.shape[0] for x in X]
    length_frequencies = Counter(lengths)
    return sorted(length_frequencies, key=lambda x: x)

def get_record(path):
    s, a = load_example(os.path.join(path))
    sample, symbol = filter_annotations(a.sample,
                                        a.symbol)
    return s[0], sample


def save_log(log, path):
    """
        Saves a log to a path.

        Parameters
        ----------
        log: list[str]
            List of log entries.
        path:
            Path to save the log to, includes the name.
    """
    with open(path, 'w+') as text_file:
        text_file.write('\n'.join(log))

def parse_log(path):
    """
        Parses a log loaded from a path.
        Extracts values of training and validation loss, and validation accuracy
        over epochs.

        Parameters
        ----------
        path:
            Path to save the log to, includes the name.

        Returns
        -------
        log: tuple[list[float], list[float], list[float]]
            Values of training and validation loss, and validation accuracy
            over epochs.

    """
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

if __name__ == '__main__':
    x, y = get_mitbih()
    print(Counter(y))
