import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import random
import csv
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define the neural network architecture (PyTorch version)
class PyTorchNetwork(nn.Module):
    def __init__(self, sizes):
        super(PyTorchNetwork, self).__init__()
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
        self.network = nn.ModuleList(self.layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.network[:-1]):
            x = self.sigmoid(layer(x))
        x = self.network[-1](x)  # No activation for output layer
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)

# Training configurations
configs = [
    [[784, 256, 128, 10], 30, 32, 0.1],  # baseline
    [[784, 256, 128, 10], 30, 32, 0.01],  # lower lr
    [[784, 256, 128, 10], 30, 32, 0.5],   # higher lr
    [[784, 256, 128, 10], 5, 32, 0.1],    # fewer tests for overfitting
    [[784, 256, 128, 10], 50, 32, 0.1],   # convergance
    [[784, 256, 128, 10], 30, 16, 0.1],   # smaller batch size
    [[784, 256, 128, 10], 30, 128, 0.1],  # larger batch size
    [[784, 64, 32, 10], 20, 32, 0.1],     # smaller network
    [[784, 256, 128, 64, 10], 30, 32, 0.1] # larger network
]

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

# Training loop for PyTorch model with MSE loss
def train_and_evaluate(config, train_loader, test_loader):
    net = PyTorchNetwork(config[0]).to(device)  # Initialize network with given layers and move to GPU if available
    criterion = nn.MSELoss()  # MSE Loss for training
    optimizer = optim.SGD(net.parameters(), lr=config[3])  # Stochastic Gradient Descent optimizer

    epoch_outputs = []
    accuracies = []  # To store accuracy per epoch for graphing
    for epoch in range(config[1]):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU

            # Flatten input
            inputs = inputs.view(-1, 784)  # Flatten the MNIST images

            # Convert labels to one-hot encoding
            one_hot_labels = torch.zeros(labels.size(0), 10).to(device)  # Ensure one-hot labels are on the same device
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1.0)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Calculate the loss
            loss = criterion(outputs, one_hot_labels)
            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{config[1]}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {epoch_acc:.4f}")
        accuracies.append(epoch_acc)  # Save accuracy for each epoch
        epoch_outputs.append([epoch, correct, total])

    # Testing the network
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU
            inputs = inputs.view(-1, 784)  # Flatten the MNIST images
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = correct / total
    print(f"Final Test Accuracy: {final_acc:.4f}")

    # Generate accuracy graph
    plot_graph(config, accuracies)

    return epoch_outputs, final_acc

# Generate and save accuracy graph
def plot_graph(config, accuracies):
    # Create the graph of accuracy over epochs
    plt.plot(range(len(accuracies)), accuracies, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over Epochs - Layers: {config[0]} - LR: {config[3]}")
    
    # Save the graph as a PNG file
    file_name = os.path.join("graphs", f"config_{'_'.join(map(str, config[0]))}_{config[1]}_{config[2]}_{config[3]}.png")
    plt.savefig(file_name)
    plt.close()

# Store results in CSV file
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Layers", "Epochs", "Mini-Batch Size", "Learning Rate", "Final Accuracy", "Epoch Outputs [#, # Correct, Total]"])
    for config in configs:
        epoch_outputs, final_acc = train_and_evaluate(config, train_loader, test_loader)
        
        # Format the epoch outputs
        epoch_output_str = " | ".join([f"Epoch {epoch+1}: {correct}/{total}" for epoch, correct, total in epoch_outputs])
        
        writer.writerow([config[0], config[1], config[2], config[3], final_acc, epoch_output_str])