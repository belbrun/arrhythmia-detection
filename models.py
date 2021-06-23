from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data import filter_beats, pad_beats, get_mitbih, map_annotations, denoise
from torchmetrics import Accuracy
from ignite.metrics import Fbeta, Accuracy
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
import torch.nn as nn
import torch
import numpy as np


# use GPU for training if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Baseline:
    """
        Baseline model class.
        Wrap svm.SVC in a structure that fits custom model interface.

        Atributes
        ---------
        kernel: str
            Specifies the kernel of the SVM.
        n_features: int
            Specifies number of features used in the model.
        use_preprocessing: bool
            Flag to determine weather to use preprocessing.
    """
    def __init__(self, kernel, n_features, use_preprocessing=True):
        self.svm = SVC(class_weight='balanced', kernel=kernel)
        self.n_features = n_features
        self.use_preprocessing = use_preprocessing

    def preprocess(self, X, y):
        """
            Prepares input data for the SVM.
            Filters beats to length of number of features of the SVM and padds
            them.

            Parameters
            ----------
            X: list[numpy.ndarray]
                List of examples.
            y: list[str]
                List of annotaton symbols.

            Returns
            -------
            X: numpy.ndarray
                Examples prepared for SVM.
            y: np.ndarray
                Annotation values.
        """
        X, y = filter_beats(X, y, self.n_features)
        #X = denoise(X)
        return pad_beats(X, self.n_features), map_annotations(y)

    def fit(self, X, y):
        """
            Fits SVM to data.

            Parameters
            -------
            X: list[numpy.ndarray]
                List of examples.
            y: list[str]
                List of annotaton symbols.

        """
        print('preparing')
        if self.use_preprocessing:
            X, y = self.preprocess(X, y)
        print(X.shape)
        print('fitting')
        self.svm.fit(X, y)


    def test(self, X, y):
        """
        Tests the SVM model and print a classification report.

        Parameters
        -------
        X: list[numpy.ndarray]
            List of examples.
        y: list[str]
            List of annotaton symbols.
        """
        if self.use_preprocessing:
            X, y = self.preprocess(X, y)
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
                 weight):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, n_classes)
        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
        self.softmax = nn.Softmax(dim=1)
        self.precision = Precision(average=False)
        self.recall = Recall(average=False)
        self.f1 = Fbeta(beta=1, average=False, precision=self.precision,
                        recall=self.recall)
        self.accuracy = Accuracy()
        self.metrics = [self.precision, self.recall, self.f1, self.accuracy]
        self.to(device)

    def forward(self, x):
        """
            Forward pass thru the model.

            Parameters
            ----------
            x: torch.Tensor
                Input batch of examples.

            Returns
            -------
            x: torch.Tensor
                Output values for the input batch (not class probabilities).
        """
        batch_size = x.size()[1]
        #_, (h, _) = self.rnn(x)
        _, h = self.rnn(x)

        x = self.fc1(h[-1])
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def train_model(self, batch, optimizer):
        """
            Execute training procedure for one batch.

            Parameters
            ----------
            batch: tuple(torch.Tensor, torch.Tensor)
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

        x, y = batch
        y_p = self(x)

        #print(y_p.size())
        loss = self.loss(y_p, y)

        loss.backward()
        optimizer.step()

        return loss.to('cpu').detach().item()

    def classify(self, x):
        """
            Forward pass thru the model with softmax layer at the end.

            Parameters
            ----------
            x: torch.Tensor
                Input batch of examples.

            Returns
            -------
            x: torch.Tensor
                Class probabilities for the input batch.
        """
        return self.softmax(self(x))

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

        x, y = batch
        y_p = self(x)
        loss = self.loss(y_p, y)

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
            x, y = batch
            y_p = self.classify(x)
            for metric in self.metrics:
                metric.update((y_p, y))
        return [metric.compute() for metric in self.metrics]





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
