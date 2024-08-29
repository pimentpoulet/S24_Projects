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
    target_transform=None
)

test_data = tv.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

class_names = train_data.classes


""" DATA LOADER """

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


""" AGNOSTIC CODE """

device = "cuda" if t.cuda.is_available() else "cpu"


""" MODEL CLASS """


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
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: t.Tensor):
        return self.layer_stack(x)


t.manual_seed(37)
model_1 = CNNfft(input_shape=784,
                 hidden_units=10,
                 output_shape=len(class_names)).to(device)


""" SETUP A LOSS FUNCTION AND OPTIMIZER """

loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(params=model_1.parameters(),
                        lr=0.1)


""" TRAINING AND TESTING LOOP """

t.manual_seed(42)
train_time_start_on_gpu = timer()

EPOCHS = 3
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n------")

    # call training function
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)

    # call testing function
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)


""" AI DURATION """

# calculate training time
train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=str(next(model_1.parameters()).device))

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)
