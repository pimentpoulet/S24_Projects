from chapter_1_1_pytorch_workflow_fundamentals import *
from pathlib import Path


# output the name of the model and the parameters it contains : quick and detailed view
# print(f"{LinearRegressionModel_0}\n\n{list(LinearRegressionModel_00.parameters())}\n\n{LinearRegressionModel_00.state_dict()}")
print()


""" MAKING PREDICTIONS USING 'TORCH.INFERENCE_MODE() """

with torch.inference_mode():           # torch.inference_mode() turns off the parameter gradient tracking from above
    y_preds = LinearRegressionModel_00(X_test)    # similar to torch.no_grad(), but more recent


""" TRAIN MODEL """

print("-- TRAIN MODEL --")

# a way to measure how poor or how wrong your models predictions are is to use a Loss function
# may also be called Cost function or Criterion in different areas
# the lower the loss function output the better

# THINGS WE NEED TO TRAIN :

# 1. Loss Function.
# 2. Optimizer : uses the loss of a model and adjusts the model's parameters
# (e.g. weight & bias).to improve the Loss Function

# WE NEED :

# 1. A Training Loop
# 2. A Testing Loop


""" SETUP A LOSS FUNCTION & OPTIMIZER """

loss_fn = nn.L1Loss()                                       # Mean Absolute Error
optimizer = torch.optim.SGD(params=LinearRegressionModel_00.parameters(),    # Stochastic Gradient Descent
                            lr=0.0001)                        # learning rate is the most important hyperparameter you can set
print()


""" SETUP A TRAINING LOOP """

# 0. Loop through the data
# 1. Forward pass (this involves data through our model's forward() method) - also called forward propagation
# 2. Calculate the loss (compare forward pass predictions to ground truth labels)
# 3. Optimizer Zero Grad
# 4. Loss backward - move backward through the network to calculate the gradients of each of the parameters with respect to the loss ('backpropagation')
# 5. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss ('gradient descent')

torch.manual_seed(42)

# an epoch is one loop through the data
epochs = 251

# track different values
epoch_count = []
loss_values = []
test_loss_values = []

# 0. Loop through the data
for epoch in range(epochs):

    # set the model to training mode
    LinearRegressionModel_00.train()    # train mode sets all parameters that require gradients to require gradients

    # 1. Forward pass on train data
    y_preds_new = LinearRegressionModel_00(X_train)

    # 2. Calculate the loss on train data
    loss = loss_fn(y_preds_new, y_train)    # predictions first, then training

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()


    """ TESTING LOOP """

    # put the model in evaluation mode
    LinearRegressionModel_00.eval()

    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = LinearRegressionModel_00(X_test)

        # 2. Calculate the loss on test data
        test_loss = loss_fn(test_pred, y_test)

        # print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
            # print(f"{model_0.state_dict()}\n")

        epoch_count.append(epoch)
        loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())

with torch.inference_mode():
    y_preds_new = LinearRegressionModel_00(X_test)

# print(f"\n{y_preds_new.squeeze(dim=1)}\n")


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


# print(f"\n{LinearRegressionModel_00.state_dict()}")
plot_predictions(predictions=(y_preds, y_preds_new))
plot_statistics()
print()


""" SAVE AND LOAD A MODEL """

# SAVE THE MODEL:

# 1. Create a model directory
MODEL_PATH = Path("models - State_dicts")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create a model save path
MODEL_NAME = "2024_01_LinearRegressionModel_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model
torch.save(obj=LinearRegressionModel_00.state_dict(),
           f=MODEL_SAVE_PATH)

# LOAD THE MODEL:

# To load in a saved state_dict(), we need to instantiate a new instance of our model class
loaded_LinearRegressionModel_00 = LinearRegressionModel_0()

# Load the saved state_dict() of LinearRegressionModel_0 (this will update the new instance with updated parameters)
loaded_LinearRegressionModel_00.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# WE CAN TEST IT WITH PREDICTIONS :
loaded_LinearRegressionModel_00.eval()
with torch.inference_mode():
    loaded_LinearRegressionModel_0_preds = loaded_LinearRegressionModel_00(X_test)

# print(loaded_LinearRegressionModel_0_preds)

# Compare loaded preds with original preds
# print(y_preds_new == loaded_LinearRegressionModel_0_preds)    # --> TRUE, TRUE, TRUE, etc.
