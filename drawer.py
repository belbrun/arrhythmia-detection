import os
import wfdb
import data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ecg_plot


def plot_beat(beat):
    t = np.arange(beat.shape[0])
    data = np.stack([t, beat[:, 0], beat[:, 1]])
    data = pd.DataFrame(data.transpose(), columns=['t', 'lead1', 'lead2'])
    sns.lineplot(data=data, x='t', y='lead1').plot()
    #ecg_plot.plot(beat.transpose(), sample_rate=500, title = 'ECG')
    #plt.show()

def plot_record(path, sampto=None):
    record = wfdb.rdrecord(path, sampto=sampto)
    annotation = wfdb.rdann(path, 'atr', sampto=sampto)
    wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4), ecg_grids='all')

def plot_training(train_loss, valid_loss, valid_acc):
    e = range(len(train_loss))
    fig, axs = plt.subplots(3)
    axs[0].plot(e, train_loss)
    axs[1].plot(e, valid_loss)
    axs[2].plot(e, valid_acc)
    plt.show()




if __name__ == '__main__':
    t, v, a = data.parse_log('state_dicts/log1.1.txt')
    plot_training(t, v, a)
    dataset_path = os.path.join('dataset', '12lead')
    diagnostics = load_diagnostics(dataset_path)
    print(Counter(diagnostics['Rhythm']))
    record = load_recording(os.path.join(dataset_path, 'ECGData', diagnostics['FileName'][5]+'.csv'))
    plot_beat(record)
