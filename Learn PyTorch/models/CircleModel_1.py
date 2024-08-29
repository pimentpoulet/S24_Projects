import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from pathlib import Path

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

# make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

# view the first example of features (X) and labels (y)
X_sample = X[0]

y_sample = y[0]

# visualize the data
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap="viridis")
plt.show()


""" TURN DATA INTO TENSORS AND CREATE TRAIN/TEST SPLITS """

X = t.from_numpy(X).type(t.float)
y = t.from_numpy(y).type(t.float)

# split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,  # 20% will be testing
                                                    random_state=42)


""" SEND DATA TO DEVICE """

# make agnostic code
device = "cuda" if t.cuda.is_available() else "cpu"

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)


""" BUILDING A MODEL """

# 1. Setup agnostic code
# 2. Construct a model
# 3. Define a loss function
# 4. Create a training and testing loop


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


CircleModel_01 = CircleModel_1()
CircleModel_01.to(device)


""" SETUP A LOSS FUNCTION AND OPTIMIZER """

loss_fn = nn.BCEWithLogitsLoss()
optimizer = t.optim.SGD(params=CircleModel_01.parameters(),
                        lr=0.05)


""" LOOP ZONE """

t.manual_seed(42)
t.cuda.manual_seed(42)

epochs = 3001

epoch_count = []
loss_values = []
test_loss_values = []

# training and testing loop
for epoch in range(epochs):

    # set the model to training mode
    CircleModel_01.train()

    # 1. Forward pass on train data
    y_logits = CircleModel_01(X_train).squeeze()
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
    CircleModel_01.eval()

    with t.inference_mode():

        # PREDICTIONS

        # 1. Forward pass on test data
        test_logits = CircleModel_01(X_test).squeeze()
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
plot_decision_boundary(CircleModel_01, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(CircleModel_01, X_test, y_test)

plt.show()


""" SAVE & LOAD """

# SAVE THE MODEL:

# 1. Create a model directory
MODEL_PATH = Path("../models - State_dicts")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create a model save path
MODEL_NAME = "2024_01_CircleModel_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dct
t.save(obj=CircleModel_01.state_dict(),
           f=MODEL_SAVE_PATH)

# LOAD THE MODEL:

# instantiate a new instance of our model class
loaded_CircleModel_01 = CircleModel_1()

# Load the saved state_dict() of LinearRegressionModel_1 (this will update the new instnce with updated parameters)
loaded_CircleModel_01.load_state_dict(t.load(f=MODEL_SAVE_PATH))

# put the loaded model on the GPU :
loaded_CircleModel_01.to(device)


""" COMPARE PREDICTIONS """

with t.inference_mode():
    loaded_CircleModel_01 = loaded_CircleModel_01(X_test)
