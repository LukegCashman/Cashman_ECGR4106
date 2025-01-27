# SECTION 0: Initialize libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# SECTION 1: Problem 1.a multilayer perceptron for Cifar10 dataset 
# Download the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the datasets
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model with 3 hidden layers
model = models.Sequential()

# Flatten the 32x32x3 input images into a 1D vector of 3072 features
model.add(layers.Flatten(input_shape=(32, 32, 3)))

# Create hidden & output layers
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Implement training model
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

# Plot training accuracy & validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')

# Add titles and labels
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot for training accuracy & validation accuracy
plt.show()

# Calculate training loss & validation loss
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')

# Add titles and labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plot for training loss & validation loss
plt.show()

# Evaluate model on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate precision, recall, F1 score, and cnf
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(10))
conf_matrix_display.plot(cmap=plt.cm.Blues)

# Print precision, recall, and F1 score
print(f'1a Precision: {precision:.4f}')
print(f'1a Recall: {recall:.4f}')
print(f'1a F1 Score: {f1:.4f}')

# Show confusion matrix plot
plt.title("Confusion Matrix")
plt.show()

# SECTION 2: Problem 2.a expanded multilayer perceptron for Cifar10 dataset 
# Download the CIFAR-10 dataset
(x_trainb, y_trainb), (x_testb, y_testb) = cifar10.load_data()

# Normalize the datasets
x_trainb = x_trainb.astype('float32') / 255.0
x_testb = x_testb.astype('float32') / 255.0

# Encode labels
y_trainb = to_categorical(y_trainb, 10)
y_testb = to_categorical(y_testb, 10)

# Define the model with 3 hidden layers
modelb = models.Sequential()

# Flatten the 32x32x3 input images into a 1D vector of 3072 features
modelb.add(layers.Flatten(input_shape=(32, 32, 3)))

# Create hidden & output layers
modelb.add(layers.Dense(512, activation='relu'))
modelb.add(layers.Dense(256, activation='relu'))
modelb.add(layers.Dense(128, activation='relu'))
modelb.add(layers.Dense(64, activation='relu'))
modelb.add(layers.Dense(10, activation='softmax'))

# Compile the model
modelb.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Implement training model
historyb = modelb.fit(x_trainb, y_trainb, epochs=20, batch_size=64, validation_data=(x_testb, y_testb))

# Plot training accuracy & validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(historyb.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(historyb.history['val_accuracy'], label='Validation Accuracy', color='red')

# Add titles and labels
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot for training accuracy & validation accuracy
plt.show()

# Calculate training loss & validation loss
plt.plot(historyb.history['loss'], label='Training Loss', color='blue')
plt.plot(historyb.history['val_loss'], label='Validation Loss', color='red')

# Add titles and labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plot for training loss & validation loss
plt.show()

# Evaluate model on the test set
y_predb = modelb.predict(x_testb)
y_pred_classesb = np.argmax(y_predb, axis=1)
y_true_classesb = np.argmax(y_testb, axis=1)

# Calculate precision, recall, F1 score, and cnf
precisionb = precision_score(y_true_classesb, y_pred_classesb, average='weighted')
recallb = recall_score(y_true_classesb, y_pred_classesb, average='weighted')
f1b = f1_score(y_true_classesb, y_pred_classesb, average='weighted')
conf_matrixb = confusion_matrix(y_true_classesb, y_pred_classesb)
conf_matrix_displayb = ConfusionMatrixDisplay(conf_matrixb, display_labels=np.arange(10))
conf_matrix_displayb.plot(cmap=plt.cm.Blues)

# Print precision, recall, and F1 score
print(f'2a Precision: {precisionb:.4f}')
print(f'2a Recall: {recallb:.4f}')
print(f'2a F1 Score: {f1b:.4f}')

# Show confusion matrix plot
plt.title("Confusion Matrix")
plt.show()

# SECTION 3: Problem 2.a
# Load the dataset from a CSV file
df2 = pd.read_csv('Housing.csv')
print(df2.head())
print(df2.shape)

x2 = df2.drop(['price','mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus'], axis=1).values
y2 = df2['price'].values

# Standardize input
object = StandardScaler()
x2 = object.fit_transform(x2)

# Split the data into train and validation sets
x_train2, x_val2, y_train2, y_val2 = train_test_split(x2, y2, test_size=0.2, random_state=42)
x_train2 = torch.tensor(x_train2, dtype=torch.float32)
y_train2 = torch.tensor(y_train2, dtype=torch.float32)
x_val2 = torch.tensor(x_val2, dtype=torch.float32)
y_val2 = torch.tensor(y_val2, dtype=torch.float32)

# Create TensorDatasets and DataLoaders for train and validation sets
train_dataset2 = TensorDataset(x_train2, y_train2)
val_dataset2 = TensorDataset(x_val2, y_val2)
train_loader2 = DataLoader(dataset=train_dataset2, batch_size=32, shuffle=True)
val_loader2 = DataLoader(dataset=val_dataset2, batch_size=32, shuffle=False)

#Definition of network model class
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Model, loss function, and optimizer
# Initialize the network
model2 = RegressionNet()
criterion = nn.MSELoss()
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

train_loss_list2 = []
val_loss_list2 = []

num_epochs = 40
for epoch in range(num_epochs):
    # Training phase
    model2.train()
    train_loss2 = 0.0
    for inputs, targets in train_loader2:
        optimizer2.zero_grad()  # Clear existing gradients
        outputs = model2(inputs)  # Forward pass
        loss2 = criterion(outputs, targets)  # Compute loss
        loss2.backward()  # Backward pass (compute gradients)
        optimizer2.step()  # Update model parameters
        train_loss2 += loss2.item() * inputs.size(0)  # Accumulate the loss

    # Calculate average training loss
    train_loss2 /= len(train_loader2.dataset)
    train_loss_list2.append(train_loss2)

    # Validation phase
    model2.eval()
    val_loss2 = 0.0
    val_total2 = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader2:
            outputs = model2(inputs)  # Forward pass
            loss2 = criterion(outputs, targets)  # Compute loss
            val_loss2 += loss2.item() * inputs.size(0)  # Accumulate the loss
            val_total2 += ((outputs - targets) ** 2).sum().item()  # Accumulate squared errors

    # Calculate average validation loss (MSE) and RMSE
    val_loss2 /= len(val_loader2.dataset)
    val_loss_list2.append(val_loss2)
    rmse2 = np.sqrt(val_total2 / len(val_loader2.dataset))

    # Print training and validation results
    print(f'Epoch[{epoch+1}/{num_epochs}], Train Loss: {train_loss2:.4f}, Validation Loss: {val_loss2:.4f}, Validation RMSE: {rmse2:.4f}')

# Plotting training and validation loss
plt.plot(train_loss_list2, label='Training Loss')
plt.plot(val_loss_list2, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Print final RMSE
print(f'Final Validation RMSE: {rmse2:.4f}')

# Evaluation loop
model2.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in val_loader2:
        outputs = model2(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate and print the accuracy
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# SECTION 4: Problem 2.b
# Load the dataset from a CSV file
df2b = pd.read_csv('Housing.csv')
df2b = pd.get_dummies(df2b,columns=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus'])
df2b = df2b.replace({True: 1, False: 0})
print(df2b.head())
print(df2b.shape)

x2b = df2b.drop(['price'], axis=1).values
y2b = df2b['price'].values

# Standardize input
object = StandardScaler()
x2b = object.fit_transform(x2b)

# Split the data into train and validation sets
x_train2b, x_val2b, y_train2b, y_val2b = train_test_split(x2b, y2b, test_size=0.2, random_state=42)
x_train2b = torch.tensor(x_train2b, dtype=torch.float32)
y_train2b = torch.tensor(y_train2b, dtype=torch.float32)
x_val2b = torch.tensor(x_val2b, dtype=torch.float32)
y_val2b = torch.tensor(y_val2b, dtype=torch.float32)

# Create TensorDatasets and DataLoaders for train and validation sets
train_dataset2b = TensorDataset(x_train2b, y_train2b)
val_dataset2b = TensorDataset(x_val2b, y_val2b)
train_loader2b = DataLoader(dataset=train_dataset2b, batch_size=32, shuffle=True)
val_loader2b = DataLoader(dataset=val_dataset2b, batch_size=32, shuffle=False)

# Definition of network model class
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Model, loss function, and optimizer
# Initialize the network
model2b = RegressionNet()
criterion = nn.MSELoss()
optimizer2b = optim.SGD(model2b.parameters(), lr=0.01)

train_loss_list2b = []
val_loss_list2b = []

num_epochs = 40
for epoch in range(num_epochs):
    # Training phase
    model2b.train()
    train_loss2b = 0.0
    for inputs, targets in train_loader2b:
        optimizer2b.zero_grad()  # Clear existing gradients
        outputs = model2b(inputs)  # Forward pass
        loss2b = criterion(outputs, targets)  # Compute loss
        loss2b.backward()  # Backward pass (compute gradients)
        optimizer2b.step()  # Update model parameters
        train_loss2b += loss2b.item() * inputs.size(0)  # Accumulate the loss

    # Calculate average training loss
    train_loss2b /= len(train_loader2b.dataset)
    train_loss_list2b.append(train_loss2b)

    # Validation phase
    model2b.eval()
    val_loss2b = 0.0
    val_total2b = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader2b:
            outputs = model2b(inputs)  # Forward pass
            loss2b = criterion(outputs, targets)  # Compute loss
            val_loss2b += loss2b.item() * inputs.size(0)  # Accumulate the loss
            val_total2b += ((outputs - targets) ** 2).sum().item()  # Accumulate squared errors

    # Calculate average validation loss (MSE) and RMSE
    val_loss2b /= len(val_loader2b.dataset)
    val_loss_list2b.append(val_loss2b)
    rmse2b = np.sqrt(val_total2b / len(val_loader2b.dataset))

    # Print training and validation results
    print(f'Epoch[{epoch+1}/{num_epochs}], Train Loss: {train_loss2b:.4f}, Validation Loss: {val_loss2b:.4f}, Validation RMSE: {rmse2b:.4f}')

# Plotting training and validation loss
plt.plot(train_loss_list2b, label='Training Loss')
plt.plot(val_loss_list2b, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Print final RMSE
print(f'Final Validation RMSE: {rmse2b:.4f}')

# Evaluation loop
model2b.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in val_loader2b:
        outputs = model2b(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate and print the accuracy
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# SECTION 5: Problem 2.c
# Create TensorDatasets and DataLoaders for train and validation sets
train_dataset2c = TensorDataset(x_train2b, y_train2b)
val_dataset2c = TensorDataset(x_val2b, y_val2b)
train_loader2c = DataLoader(dataset=train_dataset2c, batch_size=32, shuffle=True)
val_loader2c = DataLoader(dataset=val_dataset2c, batch_size=32, shuffle=False)

# Definition of network model class
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

# Model, loss function, and optimizer
# Initialize the network
model2c = RegressionNet()
criterion = nn.MSELoss()
optimizer2c = optim.SGD(model2c.parameters(), lr=0.01)

train_loss_list2c = []
val_loss_list2c = []

num_epochs = 40
for epoch in range(num_epochs):
    # Training phase
    model2c.train()
    train_loss2c = 0.0
    for inputs, targets in train_loader2c:
        optimizer2c.zero_grad()  # Clear existing gradients
        outputs = model2c(inputs)  # Forward pass
        loss2c = criterion(outputs, targets)  # Compute loss
        loss2c.backward()  # Backward pass (compute gradients)
        optimizer2c.step()  # Update model parameters
        train_loss2c += loss2c.item() * inputs.size(0)  # Accumulate the loss

    # Calculate average training loss
    train_loss2c /= len(train_loader2c.dataset)
    train_loss_list2c.append(train_loss2c)

    # Validation phase
    model2c.eval()
    val_loss2c = 0.0
    val_total2c = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader2c:
            outputs = model2c(inputs)  # Forward pass
            loss2c = criterion(outputs, targets)  # Compute loss
            val_loss2c += loss2c.item() * inputs.size(0)  # Accumulate the loss
            val_total2c += ((outputs - targets) ** 2).sum().item()  # Accumulate squared errors

    # Calculate average validation loss (MSE) and RMSE
    val_loss2c /= len(val_loader2c.dataset)
    val_loss_list2c.append(val_loss2c)
    rmse2c = np.sqrt(val_total2c / len(val_loader2c.dataset))

    # Print training and validation results
    print(f'Epoch[{epoch+1}/{num_epochs}], Train Loss: {train_loss2c:.4f}, Validation Loss: {val_loss2c:.4f}, Validation RMSE: {rmse2c:.4f}')

# Plotting training and validation loss
plt.plot(train_loss_list2c, label='Training Loss')
plt.plot(val_loss_list2c, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Print final RMSE
print(f'Final Validation RMSE: {rmse2c:.4f}')

# Evaluation loop
model2c.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in val_loader2c:
        outputs = model2c(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate and print the accuracy
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')