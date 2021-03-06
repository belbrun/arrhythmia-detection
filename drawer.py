import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

def plot_beat(beat):
    t = np.arange(beat.shape[0])
    fig, axs = plt.subplots(2)
    axs[0].plot(t, beat[:, 0])
    axs[1].plot(t, beat[:, 1])
    plt.show()

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
