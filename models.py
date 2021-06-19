from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from ignite.metrics.confusion_matrix import ConfusionMatrix #import ClassificationReport
import torch.nn as nn
import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Baseline:

    def __init__(self, kernel):
        self.svm = SVC(class_weight='balanced', kernel=kernel)

    def fit(self, X, y):
        print(X.shape)
        print('fitting')
        self.svm.fit(X, y)


    def test(self, X, y):
        print('scoring')
        print(self.svm.score(X, y))

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, n_classes,
                 weight, n_features):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size+n_features, (hidden_size+n_features)*2)
        self.fc2 = nn.Linear((hidden_size+n_features)*2, (hidden_size+n_features)//2)
        self.fc3 = nn.Linear((hidden_size+n_features)//2, n_classes)
        self.activation = nn.Tanh()
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
        self.softmax = nn.Softmax(dim=1)
        self.metric = ConfusionMatrix(n_classes)
        self.to(device)

    def forward(self, x, f):
        x = torch.reshape(x, (x.size()[2], x.size()[1], x.size()[3]))
        batch_size = x.size()[1]
        #h0 = torch.zeros(self.num_layers, batch_size,
        #                 self.hidden_size).requires_grad_().to(device)
        #c0 = torch.zeros(self.num_layers, batch_size,
        #                 self.hidden_size).requires_grad_().to(device)
        _, (h, _) = self.rnn(x)
        #_, h = self.rnn(x)
        #print(h[-1], f[-1])
        if self.n_features > 0:
            h = torch.cat((h[-1], f[-1]), 1)
        #print('h: ', h)
        x = self.fc1(h)
        #print('x: ', x)

        x = self.activation(x)
        #print('x: ', x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

    def train_model(self, batch, optimizer):

        self.train()
        self.zero_grad()

        x, f, y = batch
        y_p = self(x, f)

        #print(torch.squeeze(y, dim=0), y_p)
        loss = self.loss(y_p, torch.squeeze(y, dim=0))

        loss.backward()
        optimizer.step()

        return loss.to('cpu').detach().item()

    def classify(self, x, f):
        return self.softmax(self(x, f))

    def evaluate_model(self, batch):

        self.eval()

        x, f, y = batch
        y_p = self(x, f)
        loss = self.loss(y_p, torch.squeeze(y, dim=0))

        return loss.to('cpu').detach().item()

    def measure(self, dataloader):
        self.metric.reset()
        for batch in dataloader:
            x, f, y = batch
            y_p = self.classify(x, f)
            #print(y_p, y)

            self.metric.update((y_p, torch.squeeze(y, dim=0)))
        return self.metric.compute().to('cpu')





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
