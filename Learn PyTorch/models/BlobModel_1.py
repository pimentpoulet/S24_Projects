import matplotlib.pyplot as plt
import pandas as pd
import torch as t

from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from functions import plot_decision_boundary, accuracy_fn

print()


""" CREATE DATA """

# set hyperparameters
NUM_CLASSES = 6
NUM_FEATURES = 2
RANDOM_SEED = 40
n_samples = 2000

# 1. create multiclass data
X_blob, y_blob = make_blobs(n_samples,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.3,              # add noise on the data
                            random_state=RANDOM_SEED)

# 2. turn data into tensors
X_blob = t.from_numpy(X_blob).type(t.float)
y_blob = t.from_numpy(y_blob).type(t.LongTensor)

# 3. create splits
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# visualise the data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap="viridis")
plt.show()
plt.close()


""" BUILDING A MODEL """

# make agnostic code
device = "cuda" if t.cuda.is_available() else "cpu"


class BlobModel_1(nn.Module):

    def __init__(self, input_features, output_features, hidden_units=8):
        """
        :param input_features: number of input features to the model
        :param output_features: number of output features (number of output classes)
        :param hidden_units: number of hidden units between layers (default 8)
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features,
                      out_features=hidden_units),
            nn.ReLU6(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.ReLU6(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model = BlobModel_1(input_features=2,
                    output_features=6)
model.to(device)


""" SEND DATA TO DEVICE """

X_blob_train, X_blob_test = X_blob_train.to(device), X_blob_test.to(device)
y_blob_train, y_blob_test = y_blob_train.to(device), y_blob_test.to(device)


""" SETUP A LOSS FUNCTION AND OPTIMIZER """

loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(params=model.parameters(),
                        lr=0.1)


""" MAKE PREDICTIONS """

with t.inference_mode():
    model.eval()
    y_logits_0 = model(X_blob_test)

# logits --> prediction probabilities --> prediction labels
# logits --> pred probs (t.softmax()) --> pred labels (t.argmax())
y_preds_0 = t.softmax(y_logits_0, dim=1)
y_preds_0 = t.argmax(y_preds_0, dim=1)


""" LOOP ZONE """

t.manual_seed(RANDOM_SEED)
t.cuda.manual_seed(RANDOM_SEED)

epochs = 651

epoch_count = []
loss_values = []
test_loss_values = []

# training and testing loop
for epoch in range(epochs):

    model.train()

    # 1. Forward pass on train data
    train_logits = model(X_blob_train)
    train_preds = t.softmax(train_logits, dim=1).argmax(dim=1)  # logits -> pred probs -> pred labels

    # 2. Calculate the loss/accuracy
    train_loss = loss_fn(train_logits,    # nn.CrossEntropyLoss expects raw logits as input
                         y_blob_train)                       # nn.CrossEntropyLoss expects prediction probabilities as input

    train_acc = accuracy_fn(y_true=y_blob_train,
                            y_pred=train_preds)

    # 3. OZG -- B -- OS
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # TESTING LOOP

    model.eval()

    with t.inference_mode():

        # PREDICTIONS

        # 1. Forward pass on test data
        test_logits = model(X_blob_test)
        test_preds = t.softmax(test_logits, dim=1).argmax(dim=1)

        # 2. Calculate the loss/accuracy on test data
        test_loss = loss_fn(test_logits,
                            y_blob_test)

        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_preds)

        # print out what's happening
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}\nTrain Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}%\n"
                  f"Test loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
            # print(f"{model_0.state_dict()}\n")

        epoch_count.append(epoch)
        loss_values.append(train_loss.to("cpu").detach().numpy())
        test_loss_values.append(test_loss.to("cpu").detach().numpy())


""" VISUALIZE THE RESULT """

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_blob_train, y_blob_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_blob_test, y_blob_test)

plt.show()

print(model.state_dict())



""" SAVE AND LOAD A MODEL """

# SAVE THE MODEL:

# 1. Create a model directory
MODEL_PATH = Path("../models - State_dicts")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create a model save path
MODEL_NAME = "2024_01_MakeBlob_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model
t.save(obj=model.state_dict(),
       f=MODEL_SAVE_PATH)

# LOAD THE MODEL:

# To load in a saved state_dict(), we need to instantiate a new instance of our model class
loaded_MakeBlob_00 = BlobModel_1(input_features=2,
                                 output_features=6)

# Load the saved state_dict() of LinearRegressionModel_0 (this will update the new instance with updated parameters)
loaded_MakeBlob_00.load_state_dict(t.load(f=MODEL_SAVE_PATH))
