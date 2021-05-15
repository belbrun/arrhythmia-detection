from data import data_loaders
from models import RNN
from torch import optim


def train_procedure(model, iterators, n_epochs, optimizer):

    train_it, valid_it, test_it = iterators
    for i in range(n_epochs):

        epoch_loss = 0
        for batch in train_it:
            loss = model.train_model(batch, optimizer)
            epoch_loss += loss
        print('Epoch {} - train: {}'.format(i, epoch_loss))

        epoch_loss = 0
        for batch in valid_it:
            loss = model.evaluate_model(batch)
            epoch_loss += loss
        print('Epoch {} - validation: {}'.format(i, epoch_loss))

    test_loss = 0
    for batch in test_it:
        loss = model.evaluate_model(batch)
        test_loss += loss
    print('Testing: {}')


input_size = 2
hidden_size = 10
num_layers = 1
dropout = 0
n_classes = 17
lr = 0.01
batch_size = 16
n_epochs = 100

def main():

    iterators = data_loaders(batch_size)
    model = RNN(input_size, hidden_size, num_layers, dropout, n_classes)
    optimizer = optim.SGD(model.parameters(), lr)
    train_procedure(model, iterators, n_epochs, optimizer)

if __name__ == '__main__':
    main()
