import torch
import torch.nn as nn


class tackleNetwork(nn.Module):
    def __init__(self, activation=nn.LeakyReLU()):
        super().__init__()
        self.activation = activation
        # Shared model components
        self.conv1 = nn.Conv2d(14, 64, 1)
        self.conv2 = nn.Conv2d(64, 200, 1)
        self.conv3 = nn.Conv2d(200, 512, 1)
        self.avg_pool1 = nn.AvgPool2d(2)
        self.flatten1 = nn.Flatten(2, 3)
        self.conv4 = nn.Conv1d(512, 200, 1)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.conv5 = nn.Conv1d(200, 64, 1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 16, 1)
        self.batchnorm3 = nn.BatchNorm1d(16)
        self.avg_pool2 = nn.AvgPool1d(2)
        self.flatten2 = nn.Flatten(1, 2)
        # Regression components
        self.regr_fc1 = nn.Linear(192, 64)
        self.regr_fc2 = nn.Linear(64, 16)
        self.regr_fc3 = nn.Linear(16, 2)
        # Clasification components
        self.class_fc1 = nn.Linear(192, 64)
        self.class_fc2 = nn.Linear(64, 11)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def shared_model(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.avg_pool1(x)
        x = self.flatten1(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.batchnorm3(x)
        x = self.avg_pool2(x)
        x = self.flatten2(x)
        return x

    def regression_model(self, x):
        x = self.regr_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.regr_fc2(x)
        x = self.activation(x)
        x = self.regr_fc3(x)
        return x

    def classification_model(self, x):
        x = self.class_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.class_fc2(x)
        x = self.activation(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        x = self.shared_model(x)
        class_out = self.classification_model(x)
        regression_out = self.regression_model(x)
        return class_out, regression_out

    def predict(self, tackleFeatures: torch.tensor):
        """
        Assumes a dataloader, e.g. batches of data are provided
        """
        self.eval()
        class_out, regression_out = self(tackleFeatures)
        return class_out, regression_out


class touchdownNetwork(nn.Module):
    def __init__(self, activation=nn.LeakyReLU()):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(14, 64, 1)
        self.conv2 = nn.Conv2d(64, 200, 1)
        self.conv3 = nn.Conv2d(200, 512, 1)
        self.avg_pool1 = nn.AvgPool2d(2)
        self.flatten1 = nn.Flatten(2, 3)
        self.conv4 = nn.Conv1d(512, 200, 1)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.conv5 = nn.Conv1d(200, 64, 1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 16, 1)
        self.batchnorm3 = nn.BatchNorm1d(16)
        self.avg_pool2 = nn.AvgPool1d(2)
        self.flatten2 = nn.Flatten(1, 2)
        self.class_fc1 = nn.Linear(192, 32)
        self.class_fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def model(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.avg_pool1(x)
        x = self.flatten1(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.batchnorm3(x)
        x = self.avg_pool2(x)
        x = self.flatten2(x)
        x = self.class_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.class_fc2(x)
        x = self.activation(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, playFeatures: torch.tensor):
        """
        Assumes a dataloader, e.g. batches of data are provided
        """
        self.eval()
        x = self(playFeatures)
        return x
