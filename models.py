from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data import filter_beats, pad_beats, get_mitbih, map_annotations, denoise
from torchmetrics import Accuracy
import torch.nn as nn
import torch
import numpy as np



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Baseline:

    def __init__(self, kernel, n_features, use_preprocessing=True):
        self.svm = SVC(class_weight='balanced', kernel=kernel)
        self.n_features = n_features
        self.use_preprocessing = use_preprocessing

    def preprocess(self, X, y):
        X, y = filter_beats(X, y, self.n_features)
        #X = denoise(X)
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
        print(classification_report(y, self.svm.predict(X)))


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, n_classes,
                 weight):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, n_classes)
        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
        self.softmax = nn.Softmax(dim=1)
        self.metric = Accuracy()
        self.to(device)

    def forward(self, x):
        batch_size = x.size()[1]
        _, (h, _) = self.rnn(x)
        #_, h = self.rnn(x, h0)

        x = self.fc1(h[-1])
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def train_model(self, batch, optimizer):

        self.train()
        self.zero_grad()

        x, y = batch
        y_p = self(x)

        #print(y_p.size())
        loss = self.loss(y_p, y)

        loss.backward()
        optimizer.step()

        return loss.to('cpu').detach().item()

    def classify(self, x):
        return self.softmax(self(x))

    def evaluate_model(self, batch):

        self.eval()

        x, y = batch
        y_p = self(x)
        loss = self.loss(y_p, y)

        return loss.to('cpu').detach().item()

    def measure(self, dataloader):
        self.metric.reset()
        for batch in dataloader:
            x, y = batch
            y_p = self.classify(x)
            #print(y_p)
            acc = self.metric(y_p, y)
        return self.metric.compute().to('cpu')





if __name__ == '__main__':
    X, y = get_mitbih()
    #X, y = X[:10000], y[:10000]
    print('split    ')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4,
                                                        stratify=y,
                                                        random_state=42)
    _, X_test, _, y_test = train_test_split(X_test, y_test,
                                            test_size=0.5,
                                            random_state=42)
    #print(X_train)
    bl = Baseline('rbf', 1000)
    bl.fit(X_train, y_train)
    bl.test(X_test, y_test)
