import torch as t
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
import os

from tqdm.auto import tqdm
from torch import nn
from PIL import Image


""" AGNOSTIC CODE """

device = "cuda" if t.cuda.is_available() else "cpu"


def accuracy_fn(y_true,
                y_pred):
    correct = t.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def plot_decision_boundary(model: t.nn.Module,
                           X: t.Tensor,
                           y: t.Tensor):
    """
    Plots decision boundaries of model predicting on X in comparison to y.
    """

    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = t.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with t.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(t.unique(y)) > 2:
        y_pred = t.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = t.round(t.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap="Blues", alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="Blues")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def print_train_time(start: float,
                     end: float,
                     device: t.device):
    """
    Prints difference between start and end time
    """

    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds.")
    return total_time


def eval_model(model: t.nn.Module,
               data_loader: t.utils.data.DataLoader,
               loss_fn: t.nn.Module,
               accuracy_fn,
               device: t.device):
    """
    Returns a dictionay containing the results of model predicting on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with t.inference_mode():
        for x, y in data_loader:

            # move data to device
            x, y = x.to(device), y.to(device)

            # make predictions
            y_pred = model(x)

            # accumulate loss and acc value *per batch*
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))

        # scale loss and acc to get average loss/acc *per batch*
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,    # only works when model is a class
            "model_loss": loss.item(),                 # turns loss into a single value
            "model_acc": acc}


def train_step_1(model: t.nn.Module,
                 data_loader: t.utils.data.DataLoader,
                 loss_fn: t.nn.Module,
                 optimizer: t.optim.Optimizer,
                 accuracy_fn,
                 device: t.device):
    """
    Performs training with model learning on data loader.
    """
    train_loss, train_acc = 0, 0

    # set model to training mode
    model.train()

    # loop through training batches
    for batch, (x, y) in enumerate(data_loader):  # equivalent to (image, label)

        # put data on target device
        x, y = x.to(device), y.to(device)

        # 1. forward pass
        y_pred = model(x)

        # 2. calculate loss *per batch*
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # 3. calculate acc *per batch*
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        # 4. optimizer zero grad
        optimizer.zero_grad()

        # 5. loss backward
        loss.backward()

        # 6. optimizer step
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


def train_step_2(model: t.nn.Module,
                 dataloader: t.utils.data.DataLoader,
                 loss_fn: t.nn.Module,
                 optimizer: t.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = t.argmax(t.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step_1(model: t.nn.Module,
                data_loader: t.utils.data.DataLoader,
                loss_fn: t.nn.Module,
                accuracy_fn,
                device: t.device):
    """
    Performs testing on model learning on data loader
    """
    test_loss, test_acc = 0, 0

    # set model to evaluating mode
    model.eval()

    # loop through testing batches
    with t.inference_mode():
        for x, y in data_loader:

            # put data on target device
            x, y = x.to(device), y.to(device)

            # 1. forward pass
            test_pred = model(x)

            # 2. calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y)

            # 3. calculate accuracy
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))

        # calculate test loss/acc average *per batch*
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    # print out what's happening
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.3f}%\n")


def test_step_2(model: t.nn.Module,
                dataloader: t.utils.data.DataLoader,
                loss_fn: t.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with t.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: t.nn.Module,
          train_dataloader: t.utils.data.DataLoader,
          test_dataloader: t.utils.data.DataLoader,
          optimizer: t.optim.Optimizer,
          loss_fn: t.nn.Module,
          epochs: int):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step_2(model=model,
                                             dataloader=train_dataloader,
                                             loss_fn=loss_fn,
                                             optimizer=optimizer)
        test_loss, test_acc = test_step_2(model=model,
                                          dataloader=test_dataloader,
                                          loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


def test_1_image(model: t.nn.Module,
                 test_dataloader: t.utils.data.DataLoader,
                 loss_fn: t.nn.Module):
    # 2. Create empty results dictionary
    test_1_results = {"test_loss": [],
                      "test_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    test_loss, test_acc = test_step_2(model=model,
                                      dataloader=test_dataloader,
                                      loss_fn=loss_fn)

    # 4. Print out what's happening
    print(
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
    )

    # 5. Update results dictionary
    test_1_results["test_loss"].append(test_loss)
    test_1_results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return test_1_results


def plot_loss_curves(results: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, "xb-", label='train_loss')
    plt.plot(epochs, test_loss, "xr-", label='test_loss')
    # plt.title('Loss')
    plt.xlabel('Epochs', fontsize=18)
    plt.legend(fontsize=16)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, ".b-", label='train_accuracy')
    plt.plot(epochs, test_accuracy, ".r-", label='test_accuracy')
    # plt.title('Accuracy')
    plt.xlabel('Epochs', fontsize=18)
    plt.legend(fontsize=16)

    plt.show()


def make_predictions(model: t.nn.Module,
                     data: list,
                     device: t.device):
    """
    Computes prediction of a model for an input data
    """
    pred_probs = []

    # send model to device
    model.to(device)

    # set model to evaluating mode
    model.eval()
    with t.inference_mode():
        for sample in data:

            # prepare the sample
            sample = t.unsqueeze(sample, dim=0).to(device)

            # forward pass (model outputs raw logits)
            pred_logits = model(sample)

            # get prediction probability
            pred_prob = t.softmax(pred_logits.squeeze(), dim=0)

            # send pred_prod to cpu
            pred_probs.append(pred_prob.cpu())

    # stack pred_probs to turn list into a tensor
    return t.stack(pred_probs)


def get_subset(image_path: str,
               data_dir: str,
               data_splits: list,
               target_classes: list,
               amount: int,
               seed: int):
    """
    separate a larger dataset into smaller subsets each containing a specific class - Works well for FOOD101
    """
    random.seed(seed)
    label_splits = {}

    # get labels
    for data_split in data_splits:
        print(f"[INFO] Creating an image split for: {data_split}...")
        label_path = data_dir/"fft"/f"{data_split}.txt"
        print(data_dir)
        print(label_path)
        with open(label_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]

        # get random subset of target classes image ID's
        number_to_sample = round(amount*len(labels))
        print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
        sampled_images = random.sample(labels, k=number_to_sample)

        # apply full paths
        image_paths = [pathlib.Path(str(image_path/sample_image) + ".jpg") for sample_image in sampled_images]
        label_splits[data_split] = image_paths

    return label_splits


def plot_transformed_image(image_paths: list,
                           transform: torchvision.transforms,
                           n: int):
    """
    Selects random images from a path of images and loads/transforms
    them then plots the original vs the transformed version
    """
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.show()


def walk_through_dir(dir_path: str):
    """
    walks through dir_path returning its content
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
