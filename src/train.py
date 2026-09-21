import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from model import CNN
from data_loader import combine_datasets, prepare_data, create_dataloaders


# --- Setup: device, model, loss function, optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()  # standard loss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 20
best_accuracy = 0.0  # tracks the best test accuracy seen so far, for checkpointing
train_losses = []    # average training loss per epoch, for the loss curve plot
test_accuracies = [] # test accuracy per epoch, for the accuracy curve plot

# --- Data pipeline: load, combine, normalize/split, wrap in DataLoaders ---
combined_X, combined_y, all_classes = combine_datasets()
with open("models/class_names.txt", "w") as f:
    for class_name in all_classes:
        f.write(class_name + "\n")

X_train, y_train, X_test, y_test = prepare_data(combined_X, combined_y)
train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)


# --- Training loop: repeat for 20 epochs ---
for epoch in range(epochs):
    epoch_loss = 0.0

    # One pass through all training batches (563 batches of 64 images)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)          # forward pass
        loss = criterion(outputs, labels)  # how wrong were the predictions for this batch

        optimizer.zero_grad()  # clear gradients from the previous batch
        loss.backward()        # backpropagation: compute gradients for every weight
        optimizer.step()       # apply the computed gradients, actually update the weights

        epoch_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_losses.append(epoch_loss / len(train_loader))  # average loss over the epoch

    # --- Evaluation on the held-out test set, once per epoch ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # no training happening here, skip gradient tracking
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # highest score per image = predicted class

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

    # Only save the model when this epoch beats every previous epoch,
    # so the final saved file is the best version, not just the last one.
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), "models/cnn_handwriting_model.pth")
        print(f"Neues bestes Modell gespeichert (Test Accuracy: {test_accuracy:.2%})")

    print(f"Epoch {epoch + 1}: Test Accuracy = {test_accuracy:.2%}")
    model.train()  # switch back to training mode for the next epoch

# --- After all epochs: plot and save the training curves ---
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss über Zeit")
plt.savefig("results/training_loss.png")
plt.close()  # close the figure so it doesn't get drawn into the next plot

plt.plot(test_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy über Zeit")
plt.savefig("results/accuracy_plot.png")
plt.close()
