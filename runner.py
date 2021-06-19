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
    train = get_dataset(splits[0] + splits[1],
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

#    X = np.array(train.recordings)
#    X = X.reshape((X.shape[0], X.shape[2]))
#    f = train.features
#    X_train = np.concatenate((X[:, :500], f), axis=1)
    X_train = train.features
    y_train = train.rhythms


#    X = np.array(test.recordings)
#    X = X.reshape((X.shape[0], X.shape[2]))
#    f = test.features
#    X_test = np.concatenate((X[:, :500], f), axis=1)
    X_test = test.features

    y_test = test.rhythms

    model = Baseline('linear')
    model.fit(X_train, y_train)
    model.test(X_train, y_train)
    model.test(X_test, y_test)




model_name = 'model5tanh'
dir = 'state_dicts'

input_size = 12
hidden_size = 128
num_layers = 2
dropout = 0.1
lr = 0.001
batch_size = 8
n_epochs = 100
n_features = 13
denoised = False
merged = True # merge classes, see data class

def train():
    iterators, dataset = data_loaders(batch_size, denoised=denoised,
                                      merged=merged)
    class_weights = dataset.get_class_weights()
    model = RNN(input_size, hidden_size, num_layers, dropout,
                len(class_weights), class_weights, n_features)
    optimizer = optim.Adagrad(model.parameters(), lr)
    best_model, log = train_procedure(model, iterators, n_epochs, optimizer)
    save(best_model, join(dir, model_name + '.pt'))
    save_log(log, join(dir, 'log' + model_name[4:] + '.txt'))

def evaluate():
    iterators, dataset = data_loaders(batch_size, denoised=denoised,
                                      merged=merged)
    class_weights = dataset.get_class_weights()
    model = RNN(input_size, hidden_size, num_layers, dropout, len(class_weights),
                class_weights, n_features)
    model.load_state_dict(load(join(dir, model_name + '.pt')))
    print(model.state_dict)
    print(model.measure(iterators[2]))





def main():
    #baseline()
    print('Cuda available: ', is_available())
    #train()
    evaluate()

if __name__ == '__main__':
    main()
