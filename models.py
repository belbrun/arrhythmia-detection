from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from data import filter_beats, pad_beats, get_mitbih, map_annotations, denoise
import torch.nn as nn
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Baseline:

    def __init__(self, kernel, n_features, use_preprocessing=True):
        self.svm = SVC(class_weight='balanced', kernel=kernel)
        self.n_features = n_features
        self.use_preprocessing = use_preprocessing

    def preprocess(self, X, y):
        X, y = filter_beats(X, y, self.n_features)
        X = denoise(X)
        return pad_beats(X, self.n_features), map_annotations(y)

    def fit(self, X, y):
        print('preparing')
        if self.use_preprocessing:
            X, y = self.preprocess(X, y)
        print(X.shape)
        print('fitting')
        self.svm.fit(X, y)


    def test(self, X, y):
        if self.use_preprocessing:
            X, y = self.preprocess(X, y)
        print('scoring')
        print(self.svm.score(X, y))

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, n_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, n_classes)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.BCELoss()
        self.to(device)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).requires_grad_().to(device)
        #c0 = torch.zeros(self.num_layers, batch_size,
        #                 self.hidden_size).requires_grad_().to(device)
        _, h = self.rnn(x, h0)
        x = self.fc1(h)
        x = self.fc2(x)
        return self.activation(x)

    def train_model(self, batch, optimizer):

        self.train()
        self.zero_grad()

        x, y = batch
        y_p = self(x)
        #print(y_p)
        loss = self.criterion(y_p.view(y_p.size()[1], y_p.size()[2]), y)

        loss.backward()
        optimizer.step()

        return loss.to('cpu').detach().item()

    def evaluate_model(self, batch):

        self.eval()

        x, y = batch
        y_p = self(x)
        loss = self.criterion(y_p.view(y_p.size()[1], y_p.size()[2]), y)

        return loss.to('cpu').detach().item()

    def measure(self, dataloader):
        pass




if __name__ == '__main__':
    X, y = get_mitbih()
    #X, y = X[:10000], y[:10000]
    print('split    ')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        stratify=y)
    #print(X_train)
    bl = Baseline('rbf', 1000)
    bl.fit(X_train, y_train)
    bl.test(X_test, y_test)
