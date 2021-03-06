from data import data_loaders, save_log
from models import RNN
from torch import optim, save, load
from copy import deepcopy

def train_procedure(model, iterators, n_epochs, optimizer):

    log = []
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

    save(best_model, 'model.pt')


    """test_loss = 0
    for batch in test_it:
        print(batch)
        loss = model.evaluate_model(batch)
        test_loss += loss
    log.append('Testing: {}'.format(epoch_loss/(len(test_it)*test_it.batch_size)))

    log.append('Test accuracy: {}'.format(model.measure(test_it)))
    print(log[-2] + '\n' + log[-1])"""

    return log


input_size = 2
hidden_size = 100
num_layers = 1
dropout = 0
n_classes = 17
lr = 0.01
batch_size = 16
n_epochs = 100

def train():
    iterators, dataset = data_loaders(batch_size)
    model = RNN(input_size, hidden_size, num_layers, dropout, n_classes,
                dataset.get_class_weights())
    optimizer = optim.SGD(model.parameters(), lr)
    log = train_procedure(model, iterators, n_epochs, optimizer)
    save_log(log)

def evaluate():
    iterators, dataset = data_loaders(batch_size)
    model = RNN(input_size, hidden_size, num_layers, dropout, n_classes,
                dataset.get_class_weights())
    model.load_state_dict(load('model.pt'))
    print(model.measure(iterators[1]))

def main():
    train()
    evaluate()

if __name__ == '__main__':
    main()
