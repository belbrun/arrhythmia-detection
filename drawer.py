import os
import wfdb
import data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ecg_plot


def plot_recording(recording):
    #t = np.arange(recording.shape[0])
    #data = np.stack([t, recording[:, 0], recording[:, 1]])
    #data = pd.DataFrame(data.transpose(), columns=['t', 'lead1', 'lead2'])
    #sns.lineplot(data=data, x='t', y='lead1').plot()
    ecg_plot.plot(recording, sample_rate=500, title = 'ECG')
    plt.show()

def plot_record(record):
    sns.set_theme(style="darkgrid")
    data = {
        'Vrijednost odvoda I / mV': record,
        'Vrijeme / ms': np.linspace(0, 1000, num=4999)
            }
    sns.lineplot(data=data, x='Vrijeme / ms', y='Vrijednost odvoda I / mV')


def plot_training(train_loss, valid_loss, valid_acc):
    e = range(len(train_loss))
    fig, axs = plt.subplots(3)
    axs[0].plot(e, train_loss)
    axs[1].plot(e, valid_loss)
    axs[2].plot(e, valid_acc)
    plt.show()

def plot_counts(counts):
    #counts = [[key, counts[key]] for key in counts.keys()]
    #data = pd.DataFrame(data=counts,
    #                    columns=['class', 'count'])
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(data=counts, x='Rhythm')
    ax.set_xlabel('Razred')
    ax.set_ylabel('Broj primjera')



if __name__ == '__main__':
    #t, v, a = data.parse_log('state_dicts/log1.1.txt')
    #plot_training(t, v, a)
    #dataset_path = os.path.join('dataset', '12lead')
    #diagnostics = data.load_diagnostics(dataset_path)
    #recording = data.load_recording(os.path.join(dataset_path, 'ECGData', diagnostics['FileName'][1]+'.csv'))
    #print(recording.shape)
    #t = np.arange(recording.shape[0])
    #plt.plot(t, recording[:, 4])
    #plt.show()
    #recording = data.load_recording(os.path.join(dataset_path, 'ECGDataDenoised', diagnostics['FileName'][1]+'.csv'))
    #t = np.arange(recording.shape[0])
    #plt.plot(t, recording[:, 4])

    #counts = data.get_class_counts(os.path.join('dataset', '12lead'))
    #plot_counts(data.load_diagnostics(os.path.join('dataset', '12lead')))
    #plt.show()

    record = data.load_recording(os.path.join('dataset', '12lead', 'ECGDataDenoised', 'MUSE_20180114_080259_74000.csv'),
                                  n_leads=1)
    print(record.shape)
    plot_record(np.squeeze(record))
    plt.show()
