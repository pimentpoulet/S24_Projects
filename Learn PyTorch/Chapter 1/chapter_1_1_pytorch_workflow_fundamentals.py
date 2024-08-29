import torch
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# from torch import Tensor
from torch import nn               # IMPORTANT !!!


""" PyTorch Documentation :
https://pytorch.org/docs/stable/index.html
"""

print()


""" CREATE A DATASET WITH LINEAR REGRESSION """

# create known parameters
weight = 0.7
bias = 0.3

# create some data
start = 0
end = 10
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# print(f"{X[:10]}\n{y[:10]}\n{len(X)}\n{len(y)}")

# create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


# build a function to visualize those sets
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))

    # plot training data in blue
    plt.scatter(train_data, train_labels, c="blue", s=4, label="Training data")
    # plot test data in green
    plt.scatter(test_data, test_labels, c="red", s=4, label="Testing data")

    # plot predictions if not None
    if predictions is not None:
        plt.scatter(test_data, predictions[0], c="orange", s=4, label=f"Initial predictions")
        plt.scatter(test_data, predictions[1], c="black", s=4, label=f"Final predictions")

    # show the legend
    plt.legend(prop={"size": 14})

    # show the graph
    plt.show()


""" BUILD A LINEAR REGRESSION MODEL """

# create a linear regression model class


class LinearRegressionModel_0(nn.Module):     # almost everything inherit from nn.Module

    def __init__(self):
        super().__init__()

        """
        self.weights = nn.Parameter(torch.rand(1,
                                               requires_grad=True,
                                               dtype=torch.float))

        self.bias = nn.Parameter(torch.rand(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        """
        self.linear_layer_1 = nn.Linear(in_features=1,     # nn.Linear implements a linear regression model layer that creates the same parameters as above
                                        out_features=200)    # to each input correspond an output -> y = mx + b

        self.linear_layer_2 = nn.Linear(in_features=200,
                                        out_features=200)

        self.linear_layer_3 = nn.Linear(in_features=200,
                                        out_features=200)

        self.linear_layer_4 = nn.Linear(in_features=200,
                                        out_features=1)

    # forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x is the input data
        return self.linear_layer_4(self.linear_layer_3(self.linear_layer_2(self.linear_layer_1(x))))


""" CHECK THE CONTENT OF THE MODEL """

torch.manual_seed(42)

LinearRegressionModel_00 = LinearRegressionModel_0()

# output the name of the model and the parameters it contains : quick and detailed view
# print(f"{LinearRegressionModel_00}\n\n{list(LinearRegressionModel_00.parameters())}\n\n{LinearRegressionModel_00.state_dict()}")
# print()

# by default, the model ends up on the CPU --> needs to be put on GPU
# LinearRegressionModel_0.to(device)
# print(next(LinearRegressionModel_0.parameters()).device)
