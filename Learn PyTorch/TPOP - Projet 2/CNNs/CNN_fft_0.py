import torch as t
import torchvision as tv
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from tqdm.auto import tqdm

from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from functions import *


""" IMPORT MNIST DATASET """

train_data = tv.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None    # how do we want to transform the label/targets?
)

test_data = tv.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None    # how do we want to transform the label/targets?
)


# print(len(train_data), len(test_data))

# see the first training data
# image, label = train_data[0]
# print(train_data[0])
# print(image, label)
# print(image.shape)      # color channels | height | width

class_names = train_data.classes
# print(class_names)

# visualize the first training data
# plt.imshow(image.squeeze(), cmap="gray")    # matplotlib doesn't need the color channels, so we remove it
# plt.title(class_names[label])
# plt.show()


""" PLOT MORE IMAGES """

# t.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4

"""
for i in range(1, rows*cols+1):
    random_idx = t.randint(0, len(train_data), size=[1]).item()
    img,label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.show()
"""

# print(train_data)


""" DATA LOADER """

# setup the batch size hyperparameter
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)             # usually good to shuffle training data

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# send data to gpu
# train_dataloader = t.from_numpy(train_dataloader).type(t.float).to(device)
# test_dataloader = t.from_numpy(test_dataloader).type(t.float).to(device)

# print(train_dataloader)
# print(test_dataloader)

# print(len(train_dataloader))          # number of batches
# print(train_dataloader.batch_size)    # number of images per batch


train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_batch.shape)


t.manual_seed(42)
random_idx = t.randint(0, len(train_features_batch), size=[1]).item()

"""
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
plt.show()
"""

# print(f"Image size : {img.shape}")
# print(f"Label : {label}, label size : {label.shape}")


""" POPPING SMOKES (CREATING BASELINE MODEL) """

# create a flatten layer
flatten_model = nn.Flatten()

# get a single sample
x = train_features_batch[0]

# flatten sample
x_out = flatten_model(x)
# print(x_out.shape)


class CNNfft(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


t.manual_seed(42)

# setup model with input parameters
model_0 = CNNfft(
    input_shape=784,                   # 28x28
    hidden_units=10,                 # how many units in the hidden layer
    output_shape=len(class_names)    # 1 for every class
)

model_0.to("cpu")

# print(model_0)

# dummy_x = t.rand([1, 1, 28, 28]).to(device)
# print(model_0(dummy_x))
# print(model_0.state_dict())


""" LOSS FUNCTION AND OPTIMIZER """

# Loss function - since we're working with multi-class data, loss function will be nn.CrossEntropyLoss()
# Optimizer - t.optim.SGD()  -->  Stochastic Gradient Descent

# setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(params=model_0.parameters(),
                        lr=0.05)
# print(optimizer)

# params to track :
# 1. Model's performance
# 2. how fast it runs

# test train_time function
start_time = timer()
# some code
end_time = timer()
print_train_time(start=start_time, end=end_time)


""" TRAINING LOOP """

# 1. Loop through epochs
# 2. Loop through training batches, perform training steps, calculate the train loss *per batch*
# 3. Loop through testing batches, perform testing steps, calculate the test loss *per batch*
# 4. Print out what's happening
# 5. Time it all

# set the seed and start the timer
t.manual_seed(42)
train_time_start_on_cpu = timer()

# set the number of epochs (we'll keep this small for faster training time)
epochs = 3


""" TRAINING AND TESTING LOOP """

# testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")

    # training
    train_loss = 0

    # loop to loop through training batches
    for batch, (x, y) in enumerate(train_dataloader):    # equivalent to (image, label)

        # set model to training mode
        model_0.train()

        # 1. forward pass
        y_pred = model_0(x)

        # 2. calculate loss (per batch)
        loss = loss_fn(y_pred,y)
        train_loss += loss

        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward
        loss.backward()

        # 5. optimizer step
        optimizer.step()

        # print out what's happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples.")

    train_loss /= len(train_dataloader)

    # testing loop
    test_loss, test_acc = 0, 0
    model_0.eval()
    with t.inference_mode():
        for x_test, y_test in test_dataloader:

            # 1. forward pass
            test_pred = model_0(x_test)

            # 2. calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test)

            # 3. calculate accuracy
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        # calculate test loss average *per batch*
        test_loss /= len(test_dataloader)

        # calculate test acc average *per batch*
        test_acc /= len(test_dataloader)

    # print out what's happening
    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

# calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))
# print(next(model_0.parameters()))

model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
print(model_0_results)
