import matplotlib.pyplot as plt
import pandas as pd
import torch as t

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
from functions import plot_decision_boundary, accuracy_fn

print()


""" MAKE CLASSIFICATION DATA """

# make 1000 samples
n_samples = 3000

# create circles
X, y = make_circles(n_samples,
                    noise=0.04,
                    random_state=42)

print(f"First 10 samples of X:\n{X[:10]}\n")
print(f"First 10 samples of y:\n{y[:10]}")

# make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

# print(circles.head(10))

# visualize the data
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap="Blues")
plt.show()

# check input and output shapes
# print(X.shape, y.shape)

# view the first example of features (X) and labels (y)
X_sample = X[0]
y_sample = y[0]

# print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
# print(f"Shape for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}\n")


""" TURN DATA INTO TENSORS AND CREATE TRAIN/TEST SPLITS """

X = t.from_numpy(X).type(t.float)
y = t.from_numpy(y).type(t.float)

# print(type(X), X.dtype, y.dtype)

# split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,  # 20% will be testing
                                                    random_state=42)

# print(len(X_train), len(X_test), len(y_train), len(y_test))
# print(X_train.shape)


""" BUILDING A MODEL """

# 1. Setup agnostic code
# 2. Construct a model
# 3. Define a loss function
# 4. Create a training and testing loop

# make agnostic code
device = "cuda" if t.cuda.is_available() else "cpu"


"""
# create model
class CircleModel_0(nn.Module):

    def __init__(self):
        super().__init__()

        # create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=5)
        self.layer_2 = nn.Linear(in_features=5,
                                 out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))  # x -> layer_1 -> layer_2 -> output


# create instance of the model
model = CircleModel_0()

# transfer model to GPU
model.to(device)


# this model can be replicated using nn.Sequential()
CircleModel_00 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
"""


""" SEND DATA TO DEVICE """

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)


"""
# MAKE UNTRAINED PREDICTIONS

with t.inference_mode():
    untrained_preds = model(X_test).to(device)

# print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
# print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
# print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
# print(f"\nFirst 10 labels:\n{y_test[:10]}")


# RAW LOGITS --> PREDICTION PROBABILITIES --> PREDICTION LABELS

# our model outputs are going to be raw logits
# we can convert these logits into prediction probabilities by passing them to some kind of activation function
# (e.g. sigmoid for binary classification and softmax for multiclass classification)
# then we can convert out model's prediction probabilities to prediction labels by either rounding them or
# taking the argmax() function.

# view the first 5 outputs of the forward pass on the test data
with t.inference_mode():
    y_logits = model(X_test)[:5]

# use sigmoid activation function on our model logits to turn them into prediction probabilities
y_pred_probs = t.sigmoid(y_logits)

# find the predicted labels from the prediction probabilities
y_preds = t.round(y_pred_probs)

with t.inference_mode():
    # in full (logits -> pred probs -> pred labels)
    y_pred_labels = t.round(t.sigmoid(model(X_test)[:5]))

# check for equality
print(t.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# get rid of extra dimension
y_preds.squeeze()

# SETUP A LOSS FUNCTION AND OPTIMIZER

loss_fn = nn.BCEWithLogitsLoss()
optimizer = t.optim.SGD(params=model.parameters(),
                        lr=0.01)

# TRAIN MODEL

t.manual_seed(42)
t.cuda.manual_seed(42)

epochs = 2001

epoch_count = []
loss_values = []
test_loss_values = []

# training and testing loop
for epoch in range(epochs):

    # set the model to training mode
    model.train()

    # 1. Forward pass on train data
    y_logits = model(X_train).squeeze()
    y_pred = t.round(t.sigmoid(y_logits))  # logits -> pred probs -> pred labels

    # 2. Calculate the loss/accuracy
    train_loss = loss_fn(y_logits,  # nn.BCEWithLogitsLoss expects raw logits as input
                         y_train)  # nn.BCELoss expects prediction probabilities as input

    train_acc = accuracy_fn(y_true=y_train,
                            y_pred=y_pred)

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Perform Backpropagation on the loss
    train_loss.backward()

    # 5. Optimizer Step (perform gradient descent)
    optimizer.step()

    # TESTING LOOP

    # put the model in evaluation mode
    model.eval()

    with t.inference_mode():

        # PREDICTIONS

        # 1. Forward pass on test data
        test_logits = model(X_test).squeeze()
        test_pred = t.round(t.sigmoid(test_logits))

        # 2. Calculate the loss/accuracy on test data
        test_loss = loss_fn(test_logits,
                            y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

        # print out what's happening
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}\nTrain Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}%\n"
                  f"Test loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
            # print(f"{model_0.state_dict()}\n")

        epoch_count.append(epoch)
        loss_values.append(train_loss.to("cpu").detach().numpy())
        test_loss_values.append(test_loss.to("cpu").detach().numpy())


def plot_statistics(epochs=None,
                    loss=None,
                    test_loss=None):
    
    # Plots model's statistics
    
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


# plot_statistics()

# visualize what's going on
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)

plt.show()

"""


""" COMMON WAYS TO IMPROVE A MODEL """

# 1. Adding layers
# 2. Increase the number of hidden units
# 3. Change/Add activation functions
# 4. Change the optimization function


""" IMPROVED MODEL """


class CircleModel_1(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=32)
        self.layer_2 = nn.Linear(in_features=32,
                                 out_features=32)
        self.layer_3 = nn.Linear(in_features=32,
                                 out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model = CircleModel_1()
model.to(device)


""" SETUP A LOSS FUNCTION AND OPTIMIZER """

loss_fn = nn.BCEWithLogitsLoss()
optimizer = t.optim.SGD(params=model.parameters(),
                        lr=0.05)


""" LOOP ZONE """

t.manual_seed(42)
t.cuda.manual_seed(42)

epochs = 4001

epoch_count = []
loss_values = []
test_loss_values = []

# training and testing loop
for epoch in range(epochs):

    # set the model to training mode
    model.train()

    # 1. Forward pass on train data
    y_logits = model(X_train).squeeze()
    y_pred = t.round(t.sigmoid(y_logits))  # logits -> pred probs -> pred labels

    # 2. Calculate the loss/accuracy
    train_loss = loss_fn(y_logits,  # nn.BCEWithLogitsLoss expects raw logits as input
                         y_train)  # nn.BCELoss expects prediction probabilities as input

    train_acc = accuracy_fn(y_true=y_train,
                            y_pred=y_pred)

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Perform Backpropagation on the loss
    train_loss.backward()

    # 5. Optimizer Step (perform gradient descent)
    optimizer.step()

    # TESTING LOOP

    # put the model in evaluation mode
    model.eval()

    with t.inference_mode():

        # PREDICTIONS

        # 1. Forward pass on test data
        test_logits = model(X_test).squeeze()
        test_pred = t.round(t.sigmoid(test_logits))

        # 2. Calculate the loss/accuracy on test data
        test_loss = loss_fn(test_logits,
                            y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

        # print out what's happening
        if epoch % 500 == 0:
            print(f"Epoch: {epoch}\nTrain Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}%\n"
                  f"Test loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
            # print(f"{model_0.state_dict()}\n")

        epoch_count.append(epoch)
        loss_values.append(train_loss.to("cpu").detach().numpy())
        test_loss_values.append(test_loss.to("cpu").detach().numpy())


# visualize what's going on
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)

plt.show()
