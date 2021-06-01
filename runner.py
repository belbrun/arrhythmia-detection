from data import data_loaders, save_log, get_splits, get_dataset, load_diagnostics
from models import RNN, Baseline
from torch import optim, save, load
from torch.cuda import is_available
from copy import deepcopy
from os.path import join
from datetime import datetime
import numpy as np


def train_procedure(model, iterators, n_epochs, optimizer):

    log = [str(model.state_dict), str(optimizer),
           'Start time: ' + str(datetime.now())]
    train_it, valid_it, test_it = iterators
    max_accuracy = 0
    for i in range(n_epochs):

        epoch_loss = 0
        for batch in train_it:
            loss = model.train_model(batch, optimizer)
            epoch_loss += loss
        log.append('Epoch {} - train: {}'.format(i, epoch_loss/(len(train_it)*train_it.batch_size)))
        print(log[-1])

        epoch_loss = 0
        for batch in valid_it:
            loss = model.evaluate_model(batch)
            epoch_loss += loss
        log.append('Epoch {} - validation: {}'.format(i, epoch_loss/(len(valid_it)*valid_it.batch_size)))
        accuracy = model.measure(valid_it)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = deepcopy(model.state_dict())
        log.append('Valid accuracy: {}'.format(model.measure(valid_it)))
        print(log[-2] + '\n' + log[-1])
    log.append('End time: ' + str(datetime.now()))


    """test_loss = 0
    for batch in test_it:
        print(batch)
        loss = model.evaluate_model(batch)
        test_loss += loss
    log.append('Testing: {}'.format(epoch_loss/(len(test_it)*test_it.batch_size)))

    log.append('Test accuracy: {}'.format(model.measure(test_it)))
    print(log[-2] + '\n' + log[-1])"""

    return best_model, log


def baseline():
    dataset_path = join('dataset', '12lead')
    diagnostics = load_diagnostics(dataset_path)
    splits = get_splits(dataset_path)
    train = get_dataset(splits[0]+splits[1],
                        dataset_path,
                        diagnostics,
                        onehot=False,
                        n_leads=1,
                        denoised=True)
    test = get_dataset(splits[2],
                       dataset_path,
                       diagnostics,
                       onehot=False,
                       n_leads=1,
                       denoised=True)

    X = np.array(train.recordings)
    X = X.reshape((X.shape[0], X.shape[2]))
    f = train.features
    print(X.shape, f.shape)
    X = np.concatenate((X[:, :1000], f), axis=1)
    y = train.rhythms

    model = Baseline('linear')
    model.fit(X, y)
    model.test(X, y)




model_name = 'model1'
dir = 'state_dicts'

input_size = 2
hidden_size = 100
num_layers = 1
dropout = 0
n_classes = 17
lr = 0.01
batch_size = 100
n_epochs = 100

def train():
    iterators, dataset = data_loaders(batch_size)
    model = RNN(input_size, hidden_size, num_layers, dropout, n_classes,
                dataset.get_class_weights())
    optimizer = optim.SGD(model.parameters(), lr)
    best_model, log = train_procedure(model, iterators, n_epochs, optimizer)
    save(best_model, join(dir, model_name + '.pt'))
    save_log(log, join(dir, 'log' + model_name[4:]))

def evaluate():
    iterators, dataset = data_loaders(batch_size)
    model = RNN(input_size, hidden_size, num_layers, dropout, n_classes,
                dataset.get_class_weights())
    model.load_state_dict(join(dir, model_name + '.pt'))
    print(model.state_dict)
    print(model.measure(iterators[1]))





def main():
    baseline()
    #print('Cuda available: ', is_available())
    #train()
    #evaluate()

if __name__ == '__main__':
    main()
