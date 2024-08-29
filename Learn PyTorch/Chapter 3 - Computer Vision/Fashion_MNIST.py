import torch as t
import torchvision as tv
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# print(t.__version__)
# print(tv.__version__)


""" SETUP TRAINING DATA """

train_data = datasets.FashionMNIST(
    root="data",                       # where to download data to
    train=True,                        # do we want the training version of the datatest?
    download=True,
    transform=ToTensor(),              # how do we want to transform the data?
    target_transform=None              # how do we want to transform the label/targets?
)

test_data = datasets.FashionMNIST(
    root="data",                       # where to download data to
    train=False,                       # do we want the training version of the datatest?
    download=True,
    transform=ToTensor(),              # how do we want to transform the data?
    target_transform=None              # how do we want to transform the label/targets?
)

# print(len(train_data), len(test_data))

# see the first training data
image, label = train_data[0]
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

for i in range(1, rows*cols+1):
    random_idx = t.randint(0, len(train_data), size=[1]).item()
    img,label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
# plt.show()


""" PREPARE DATALOADER """

print(train_data, '\n\n', test_data)

# turn Train dataset into DataLoader
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=32,
                             shuffle=True)









