from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ignite.metrics.confusion_matrix import ConfusionMatrix #import ClassificationReport
from ignite.metrics import Fbeta, Accuracy
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
import torch.nn as nn
import torch


# use GPU for training if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Baseline:
    """
        Baseline model class.
        Wrap svm.SVC in a structure that fits custom model interface.

        Atributes
        ---------
        svm: svm.SVC
            SVM model.
    """
    def __init__(self, kernel):
        self.svm = SVC(class_weight='balanced', kernel=kernel)

    def fit(self, X, y):
        """
            Fits SVM to data.

            Parameters
            -------
            X: numpy.ndarray
                Input features.
            y: numpy.ndarray
                Class annotations.
        """
        print(X.shape)
        print('fitting')
        self.svm.fit(X, y)


    def test(self, X, y):
        """
        Tests the SVM model and print a classification report.

        Parameters
        -------
        X: numpy.ndarray
            Input features.
        y: numpy.ndarray
            Class annotations.
        """
        print('scoring')
        print(classification_report(y, self.svm.predict(X)))

class RNN(nn.Module):


    """
        Models an RNN model as nn.Module.

        Atributes
        ---------
        input_size: int
            Size of input features of each element of the sequence.
        hidden_size: int
            Size of RNNs hidden state.
        n_features: int
            Number of additional features used.
        num_layes: int
            Number of RNN layers.
        dropout: float
            Dropout value.
        n_classes: int
            Number of classes.
        weight: numpy.ndarray
            Class weights.
        rnn: nn.Module
            RNN.
        fc1: nn.Module
            First fully connected layer.
        fc2: nn.Module
            Second fully connected layer.
        fc3: nn.Module
            Third fully connected layer.
        activation: nn.Function
            Activation function.
        loss: nn.Function
            Loss function.
        precision: ignite.Metric
            Precision metric.
        recall: ignite.Metric
            Recall metric.
        f1: ignite.Metric
            F1 metric.
        accuracy: ignite.Metric
            Accuracy metric.
        metrics: list[Metric]
            List of all contained metrics.
    """
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
        #self.fc1 = nn.Linear(hidden_size+n_features, (hidden_size+n_features)//2)
        #self.fc2 = nn.Linear((hidden_size+n_features)//2, n_classes)


        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
        self.softmax = nn.Softmax(dim=1)
        self.precision = Precision(average=False)
        self.recall = Recall(average=False)
        self.f1 = Fbeta(beta=1, average=False, precision=self.precision,
                        recall=self.recall)
        self.accuracy = Accuracy()
        self.metrics = [self.precision, self.recall, self.f1, self.accuracy]
        self.to(device)

    def forward(self, x, f):
        """
            Forward pass thru the model.

            Parameters
            ----------
            x: torch.Tensor
                Input batch of recordings.
            f: torch. Tensor
                Input batch of additional features.

            Returns
            -------
            x: torch.Tensor
                Output values for the input batch (not class probabilities).
        """
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
        """
            Execute training procedure for one batch.

            Parameters
            ----------
            batch: tuple(torch.Tensor, torch.Tensor, torch.Tensor)
                Input batch.
            optimizer: nn.Optimizer
                Optimizer.

            Returns
            -------
            loss: float
                Loss of the model for the input batch.
        """
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
        """
            Forward pass thru the model with softmax layer at the end.

            Parameters
            ----------
            x: torch.Tensor
                Input batch of recordings.
            f: torch. Tensor
                Input batch of additional features.

            Returns
            -------
            x: torch.Tensor
                Class probabilities for the input batch.
        """
        return self.softmax(self(x, f))

    def evaluate_model(self, batch):
        """
            Execute evaluation procedure for one batch.

            Parameters
            ----------
            batch: tuple(torch.Tensor, torch.Tensor)
                Input batch.

            Returns
            -------
            loss: float
                Loss of the model for the input batch.
        """
        self.eval()

        x, f, y = batch
        y_p = self(x, f)
        loss = self.loss(y_p, torch.squeeze(y, dim=0))

        return loss.to('cpu').detach().item()

    def measure(self, dataloader):
        """
            Measure the models performance with defined metrics on a dataset.

            Parameters
            ----------
            dataloadet: torch.DataLoader
                Iterator over a dataset.

            Returns
            -------
            metrics: list[torch.Tensor]
                List of defined metrics values for every class
                (accuracy is averaged).
        """
        for metric in self.metrics:
            metric.reset()
        for batch in dataloader:
            x, f, y = batch
            y_p = self.classify(x, f)
            for metric in self.metrics:
                metric.update((y_p, torch.squeeze(y, dim=0)))
        return [metric.compute() for metric in self.metrics]





if __name__ == '__main__':
    X, y = get_mitbih()
    X, y = X[:10000], y[:10000]
    print('split    ')
    X_train, X_validtest, y_train, y_validtest = train_test_split(recording_paths,
                                                        y,
                                                        train_size=train_size,
                                                        stratify=y)
    _, X_test, _, y_test = train_test_split(x_validtest,
                                                        y_validtest,
                                                        train_size=0.5,
                                                        stratify=y_validtest)
    #print(X_train)
    bl = Baseline('rbf')
    bl.fit(X_train, y_train)
    bl.test(X_test, y_test)
