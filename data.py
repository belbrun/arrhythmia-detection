import os
import wfdb


def load_example(path):
    signal = wfdb.rdsamp(path, sampto=None)
    annotation = wfdb.rdann(path, 'atr', sampto=None)
    return record, annotation

def split_record(signal, samples):
    beats = []
    pos = samples[0]
    samples = samples[1:]
    for i in range(len(samples)):
        beat = signal[pos: sample[]]


def plot(path, sampto):
    record = wfdb.rdrecord(path, sampto=sampto)
    annotation = wfdb.rdann(path, 'atr', sampto=sampto)
    wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4), ecg_grids='all')

if __name__ == '__main__':
    plot('dataset/mit-bih/109', 3000)
    r, a = load_example('dataset/mit-bih/109')
    print(a.symbol[:100])
    print(sum(a.sample[2:]-a.sample[1:-1])/(len(a.sample)-1))
