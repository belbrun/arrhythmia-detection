import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from collections import Counter
from sklearn.model_selection import train_test_split

# use GPU for training if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class annotations
class_mapping = {'SB':0, 'SR':1, 'AFIB':2, 'ST':3, 'SVT':4, 'AF':5, 'SA':6,
                 'AT':7, 'AVNRT':8, 'AVRT':9, 'SAAWR':10}

# class annotations for merged classes
merged_mapping = {'SB':2, 'SR':3, 'AFIB':0, 'ST':1, 'SVT':1, 'AF':0, 'SA':3,
                 'AT':1, 'AVNRT':1, 'AVRT':1, 'SAAWR':1}

#merged_mapping = {['AFIB', 'AF']: 0, ['SVT', 'AT', 'SAAWR', 'ST', 'AVNTR']: 1,
#                  ['SB']: 2, ['SR', 'SI']: 3}

# feature names
feature_columns = ['PatientAge', 'Gender', 'VentricularRate', 'AtrialRate',
                   'QRSDuration', 'QTInterval', 'QTCorrected', 'RAxis', 'TAxis',
                   'QRSCount', 'QOnset', 'QOffset', 'TOffset']

class ShaoxingDataset(Dataset):

    """
        Model the Chapman-Shaoxing arrythmia dataset.

        Atributes
        ----------
        recordings: numpy.ndarray
            ECG recorings.
        features: numpy.ndarray
            Additional features.
        rhythms: numpy.ndarray
            Rhythm annotations.
    """
    def __init__(self, recordings, features, rhythms):
        self.recordings = recordings
        self.features = features
        self.rhythms = rhythms

    def __len__(self):
        """
        Return number of examples in the dataset

        Returns
        -------
        length: int
            Number of examples in the dataset.
        """
        return len(self.recordings)

    def __getitem__(self, index):

        if isinstance(index, list):
            X = torch.Tensor(self.recordings[index]).to(device)
            f = torch.Tensor(self.features[index]).to(device)
            y = torch.LongTensor(self.rhythms[index]).to(device)
            return X, f, y

        return self.recordings[index], self.features[index], self.rhythms[index]

    def get_class_weights(self):
        """
            Calculate class weights.

            Returns
            -------
            weights: numpy.ndarray
                Vector of class weights.
        """
        counts = Counter(self.rhythms.tolist() if len(self.rhythms.shape) == 1 else
                         np.argmax(self.rhythms, axis=1).tolist())
        weights = np.empty(len(counts))
        print(counts)
        for c in counts:
            weights[c] = counts[c]
        #return counts.most_common(1)[0][1]/weights
        return self.rhythms.shape[0]/(len(class_mapping.keys())*weights)

def load_diagnostics(path):
    """
        Load diagnostcs file.

        Parameters
        ----------
        path: os.path
            Dataset path.

        Returns
        -------
        diagnostics: pandas.DataFrame
            Diagnostics file contents.
    """
    return pd.read_excel(os.path.join(path, 'Diagnostics.xlsx'))


def load_recording(path, n_leads=12):
    """
        Load ECG recording.

        Parameters
        ----------
        path: os.path
            Recording path.
        n_leads: int
            Number of channels of the recording to be retured.

        Returns
        -------
        recording: np.ndarray
            Recording.
    """
    recording = pd.read_csv(path, sep=',', header=0)
    return recording.values[:, :n_leads].astype('float32')

def map_annotations(annotations, onehot=False, merged=False):
    """
        Map annotation symbols to numerical values or to one-hot encoded
        vectors.

        Parameters
        ----------
        annotations: list[str]
            Annotation symbols.
        onehot: bool
            Flag to determine if the return should be a one hot encoded vector.
        merged: bool
            Flag to determine if the annotations should be mapped to their
            merged values.

        Returns
        -------
        annotations: np.ndarray
            Array containing numerical values or one-hot encoded vectors of
            annotations.
    """
    mapping = class_mapping if not merged else merged_mapping
    if onehot:
        Y = np.zeros((len(annotations), len(set(mapping.values()))))
    else:
        Y = np.empty(len(annotations), dtype='int32')
    for i, annotation in enumerate(annotations):
        if onehot:
            Y[i, mapping[annotation]] = 1
        else:
            Y[i] = mapping[annotation]
    return Y

def map_gender(gender):
    """
        Map gender feature to numerical value.

        Parameters
        ----------
        gender: str
            Gender string.

        Returns
        -------
        gender: int
            Gender integer value.
    """
    return 0 if gender == 'MALE' else 1

def standardize_features(features):
    """
        Standardize features to achieve mean = 0, stddev = 1.

        Parameters
        ----------
        features: pandas.DataFrame
            Features.

        Returns
        -------
        features: pandas.DataFrame
            Standardized features.
    """
    #print((features-features.mean())/features.std())
    return (features-features.mean())/features.std()

def get_splits(dataset_path, include=['train', 'valid', 'test']):
    """
        Load dataset splits from pregenrated files.

        Parameters
        ----------
        dataset_path: os.path
            Dataset path.
        include: list[str]
            List of splits to include.

        Returns
        -------
        splits: list[list[str]].
            List of lists which contain names of recordings that belong to the
            split.
    """
    splits = []
    for split in include:
        with open(os.path.join(dataset_path, split+'_rec.txt'), 'r+') as text_file:
            splits.append(text_file.read().splitlines())
    return splits

def get_dataset(split, dataset_path, diagnostics, onehot=False, n_leads=12,
                denoised=False, merged=False):
    """
        Generates a dataset object given the split.

        Parameters
        ----------
        splits: list[str]].
            List which contain names of recordings that belong to the
            split.
        dataset_path: os.path
            Dataset path.
        diagnostics: pandas.DataFrame
            Diagnostics dataframe.
        onehot: bool
            Flag to decide if annotations are mapped to a one-hot encoded vector.
        n_leads: int
            Number of channels of the recordings included in the dataset.
        denoised: bool
            Flag to decide if the denoised data is loaded.
        merged: bool
            Flag to determine if the annotations should be mapped to their
            merged values.

        Returns
        -------
        dataset: ShaoxingDataset
            Dataset object.
    """
    path = os.path.join(dataset_path, 'ECGDataDenoised' if denoised else 'ECGData')
    recordings = []
    for recording_path in split:
        recordings.append(load_recording(os.path.join(path,
                                                      recording_path +'.csv'),
                                         n_leads))
        #print(recordings[-1])

    split = diagnostics.loc[diagnostics['FileName'].isin(split)]
    features = preprocess_features(split[feature_columns])
    rhythms = split['Rhythm']
    features.loc[:, 'Gender'] = features['Gender'].apply(map_gender)
    return ShaoxingDataset(np.array(recordings), features.values,
                           map_annotations(rhythms.values, onehot, merged))

def preprocess_features(features):
    """
        Preprocess features using map_gender and standardize_features

        Parameters
        ----------
        features: pandas.Dataframe
            Features.

        Returns
        -------
        features: pandas.Dataframe
            Preprocessed features.
    """
    features.loc[:, 'Gender'] = features['Gender'].apply(map_gender)
    noncategorical = list(features)
    noncategorical.remove('Gender')
    features.loc[:, noncategorical] = \
            features.loc[:, noncategorical].apply(standardize_features, axis=0)

    return features




def data_loaders(batch_size, include=['train', 'valid', 'test'],
                 n_leads=12, denoised=False, merged=False):
    """
        Generate batch iterators over train, validation and test splits of the
        dataset.
        Load the included splits of the dataset.
        Generate BatchSamplers over MITBIHDataset objects, then uses it to
        generate DataLoader objects.
        Parameters
        ----------
        batch_size: int
            Batch size.
        include: list[str]
            Splits to include.
        n_leads: int
            Number of channels of the recordings included in the dataset.
        denoised: bool
            Flag to decide if the denoised data is loaded.
        merged: bool
            Flag to determine if the annotations should be mapped to their
            merged values.

        Returns
        -------
        data_loaders: list[DataLoader]
            List of train, validation and test data loaders, respectfully.
        datasets:
            List of train, validation and test datasets, respectfully.
    """
    dataset_path = os.path.join('dataset', '12lead')
    diagnostics = load_diagnostics(dataset_path)
    dss = []
    for split in get_splits(dataset_path, include):
        dss.append(get_dataset(split, dataset_path, diagnostics,
                               denoised=denoised, merged=merged))

    samplers = [BatchSampler(RandomSampler(ds), batch_size=batch_size,
                           drop_last=False) for ds in dss]

    dls = [DataLoader(dss[i], sampler=sampler) for i, sampler in
           enumerate(samplers)]

    return dls, dss



def get_length_frequencies(X):
    lengths = [x.shape[0] for x in X]
    length_frequencies = Counter(lengths)
    return length_frequencies.sorted(key=lambda pair: pair[0])

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

def split_dataset(path, train_size=0.6):
    """
        Split dataset and generate split values.

        Parameters
        ----------
        path: os.path
            Dataset path.
        train_size: float
            Ratio of data to go to the train set, valid and test are split
            equally.

        Returns
        -------
        splits: list[list[str]].
            List of lists which contain names of recordings that belong to the
            split.
    """
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

def get_class_counts(path, ordered=False):
    diagnostics = load_diagnostics(path)
    recording_paths = diagnostics['FileName']
    rhythms = diagnostics['Rhythm']
    return Counter(rhythms)

def feature_parameters(path):
    diagnostics = load_diagnostics(path)
    features = diagnostics[feature_columns]
    description = features.describe()
    for column in description:
        print(description[column])
        print()


if __name__ == '__main__':
    dataset_path = os.path.join('dataset', '12lead', 'ECGData')
    feature_parameters(os.path.join('dataset', '12lead'))
    rec = load_recording(os.path.join(dataset_path, 'MUSE_20180111_155115_19000.csv'), 4)
    print(rec.shape, rec)
    diagnostics = load_diagnostics(os.path.join('dataset', '12lead'))
    dataset = get_dataset(get_splits(os.path.join('dataset', '12lead'), ['test'])[0], os.path.join('dataset', '12lead'), diagnostics)
    x, f, y =  dataset.__getitem__([1, 17, 32, 59, 19])
    print(x.size())
