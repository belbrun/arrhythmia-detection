import os
import wfdb
import data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ecg_plot


def plot_beat(beat):
    #t = np.arange(beat.shape[0])
    #data = np.stack([t, beat[:, 0], beat[:, 1]])
    #data = pd.DataFrame(data.transpose(), columns=['t', 'lead1', 'lead2'])
    #sns.lineplot(data=data, x='t', y='lead1').plot()
    ecg_plot.plot(beat.transpose(), sample_rate=500, title = 'ECG')
    plt.show()

def plot_record(record):
    print(record.shape)
    sns.set_theme(style="darkgrid")
    data = {
        'Vrijednost odvoda I / mV': record,
        'Vrijeme / ms': np.linspace(0, 30*60*1000, num=650000)
            }
    return sns.lineplot(data=data, x='Vrijeme / ms', y='Vrijednost odvoda I / mV')

def plot_splits(axis, positions):
    for position in positions:
        plt.axvline(position, color='red')

def plot_training(train_loss, valid_loss, valid_acc):
    sns.set_theme(style="darkgrid")
    e = range(len(train_loss))
    data = pd.DataFrame({'Epoha': e, 'Skup za treniranje': train_loss,
            'Skup za validaciju': valid_loss})
    #print(data)
    ax = sns.lineplot(x='Epoha', y='value', hue='Skup',
                      legend='full', data=pd.melt(data, ['Epoha'], var_name='Skup'))
    ax.set_xlabel('Epoha')
    ax.set_ylabel('Gubitak')

def plot_counts(annotations):
    #counts = [[key, counts[key]] for key in counts.keys()]
    #data = pd.DataFrame(data=counts,
    #                    columns=['class', 'count'])
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(data=annotations, x='annotations')
    ax.set_xlabel('Razred')
    ax.set_ylabel('Broj primjera')

def plot_lengths(lengths):
    sns.set_theme(style="darkgrid")
    ax = sns.displot(data=lengths, x='Duljina otkucaja', kde=True)
    #ax.set_xlabel('Duljina otkucaja')
    #ax.set_ylabel('Broj otkucaja')



if __name__ == '__main__':
    t, v, a = data.parse_log('state_dicts/log2.txt')
    plot_training(t, v, a)
    plt.show()
    #beat = data.get_mitbih()[0][5]
    #plot_beat(beat)

    #record, sample = data.get_record(os.path.join('dataset', 'mit-bih', '108'))
    #positions = data.get_split_positions(record, sample)
    #axis = plot_record(record[:, 0])
    #plot_splits(axis, [position*30*60*1000/650000 for position in positions])
    #plt.show()

    #_, annotations = data.get_mitbih()
    #annotations = ['Pravilan' if x == 'N' else 'Nepravilan' for x in annotations]
    #annotations = [x for x in annotations if x != 'N']
    #plot_counts({'annotations': annotations})
    #plt.show()

    #records, _ = data.get_mitbih()
    #lengths = [x for x in data.get_length_frequencies(records) if x < 2000]
    #plot_lengths({'Duljina otkucaja': lengths})
    #plt.show()
