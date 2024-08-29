import torch
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path


""" DEVICE AGNOSTIC CODE """

device = "cuda" if torch.cuda.is_available() else "cpu"


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

# create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


""" BUILD A LINEAR REGRESSION MODEL """


# create a linear regression model class
class LinearRegressionModel_1(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1,
                                      # nn.Linear implements a linear regression model layer that creates the same parameters as the first time
                                      out_features=1)  # to each input correspond an output -> y = mx + b

    # forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


""" CREATE THE MODEL """

torch.manual_seed(42)
LinearRegressionModel_01 = LinearRegressionModel_1()


""" PREDICTIONS """

with torch.inference_mode():  # torch.inference_mode() turns off the parameter gradient tracking from above
    y_pre_preds = LinearRegressionModel_01(X_test)  # similar to torch.no_grad(), but more recent


""" PUT EVERYTHING ON THE GPU """

# model
LinearRegressionModel_01.to(device)

# data
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


""" SETUP A LOSS FUNCTION & OPTIMIZER """

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=LinearRegressionModel_01.parameters(),
                            lr=0.01)


""" TRAINING LOOP """

torch.manual_seed(42)

# an epoch is one loop through the data
epochs = 2001

# track different values
epoch_count = []
loss_values = []
test_loss_values = []

# 0. Loop through the data
for epoch in range(epochs):

    # set the model to training mode
    LinearRegressionModel_01.train()

    # 1. Forward pass on train data
    y_train_pred = LinearRegressionModel_01(X_train)

    # 2. Calculate the loss on train data
    loss = loss_fn(y_train_pred, y_train)

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Perform Backpropagation on the loss
    loss.backward()

    # 5. Optimizer Step (perform gradient descent)
    optimizer.step()


    """ TESTING LOOP """

    # put the model in evaluation mode
    LinearRegressionModel_01.eval()

    with torch.inference_mode():


        """ PREDICTIONS """

        # 1. Forward pass on test data
        y_test_preds = LinearRegressionModel_01(X_test)

        # 2. Calculate the loss on test data
        test_loss = loss_fn(y_test_preds, y_test)

        # print out what's happening
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
            # print(f"{model_0.state_dict()}\n")

        epoch_count.append(epoch)
        loss_values.append(loss.to("cpu").detach().numpy())
        test_loss_values.append(test_loss.to("cpu").detach().numpy())


""" CHECK FINAL VALUES """

print(f"\n{LinearRegressionModel_01.state_dict()}")


""" FUNCTIONS """


def plot_predictions(train_data=None,
                     train_labels=None,
                     test_data=None,
                     test_labels=None,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    """
    if train_data is None:
        train_data = X_train.to("cpu")
    if train_labels is None:
        train_labels = y_train.to("cpu")
    if test_data is None:
        test_data = X_test.to("cpu")
    if test_labels is None:
        test_labels = y_test.to("cpu")
    if predictions is None:
        predictions = [y_pre_preds.to("cpu"), y_test_preds.to("cpu")]

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


def plot_statistics(epochs=None,
                    loss=None,
                    test_loss=None):
    """
    Plots model's statistics
    """
    if test_loss is None:
        test_loss = test_loss_values
    if loss is None:
        loss = loss_values
    if epochs is None:
        epochs = epoch_count

    plt.figure(figsize=(10, 7))

    # plot loss data in blue
    plt.plot(epochs, loss, label="Train loss data")
    # plot test_loss data in green
    plt.plot(epochs, test_loss, label="Test loss data")

    # show the axis names
    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    # show the title
    plt.title("Training and test loss curve")

    # show the legend
    plt.legend()

    # show the graph
    plt.show()


# plot_predictions()
plot_statistics()


""" SAVE & LOAD """

# SAVE THE MODEL:

# 1. Create a model directory
MODEL_PATH = Path("../models - State_dicts")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create a model save path
MODEL_NAME = "2024_01_LinearRegressionModel_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dct
torch.save(obj=LinearRegressionModel_01.state_dict(),
           f=MODEL_SAVE_PATH)

# LOAD THE MODEL:

# instantiate a new instance of our model class
loaded_LinearRegressionModel_01 = LinearRegressionModel_1()

# Load the saved state_dict() of LinearRegressionModel_1 (this will update the new instance with updated parameters)
loaded_LinearRegressionModel_01.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# put the loaded model on the GPU :
loaded_LinearRegressionModel_01.to(device)

""" COMPARE PREDICTIONS """
with torch.inference_mode():
    loaded_LinearRegressionModel_01_preds = loaded_LinearRegressionModel_01(X_test)
