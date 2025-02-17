# Section 0: Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Section 1: Problem 1 AlexNet Model
# Load and preprocess Cifar10 or Cifar100 dataset
#(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data() # Comment this statement and
#                                                          uncomment below statement for Cifar100
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data() # Comment this statement and
#                                                          uncomment above statement for Cifar10

x_train, x_test = x_train / 255.0, x_test / 255.0
x_val = x_train[45000:]
y_val = y_train[45000:]
x_train = x_train[:45000]
y_train = y_train[:45000]

# Define AlexNet model
def create_alexnet_model():
    model = models.Sequential([
        # First Convolutional Layer
        layers.Conv2D(64, (3, 3), strides=1, activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Second Convolutional Layer
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Third Convolutional Layer
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        
        # Flatten the output
        layers.Flatten(),
        
        # Fully Connected Layer
        layers.Dense(1024, activation='relu'),
        #layers.Dropout(0.5), # Uncomment to add dropout functionality
        
        # Output Layer
        #layers.Dense(10, activation='softmax') # Comment this statement and
        #                                         uncomment below statement for Cifar100
        layers.Dense(100, activation='softmax') # Comment this statement and
        #                                         uncomment above statement for Cifar10
    ])
    return model

# Create the model
model = create_alexnet_model()

# Compile model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=64, 
                    validation_data=(x_val, y_val), 
                    callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Plot training and validation metrics
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# SECTION 2: Problem 2 VGGNet Model
# Load Cifar10 or Cifar100 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data() # Comment this statement and
#                                                              uncomment below statement for Cifar100
#(x_train, y_train), (x_test, y_test) = cifar100.load_data() # Comment this statement and
#                                                             uncomment above statement for Cifar10

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding of labels
y_train = to_categorical(y_train, 10) # Comment these statements and
y_test = to_categorical(y_test, 10)   # uncomment below statements for Cifar100

#y_train = to_categorical(y_train, 100) # Comment these statements and
#y_test = to_categorical(y_test, 100)   # uncomment above statements for Cifar10

# VGGNet Model
def create_optimized_vggnet():
    model2 = models.Sequential()

    # Block 1
    model2.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model2.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model2.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model2.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model2.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model2.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling layer instead of Flatten
    model2.add(layers.GlobalAveragePooling2D())
    
    # Fully connected layers
    model2.add(layers.Dense(256, activation='relu'))
    model2.add(layers.Dropout(0.5)) # Uncomment for dropout functionality
    model2.add(layers.Dense(10, activation='softmax')) # Comment this statement and
    #                                                      uncomment below statement for Cifar100
    #model2.add(layers.Dense(100, activation='softmax')) # Comment this statement and
    #                                                     uncomment above statement for Cifar10

    return model2

# Create VGGNet model
model2 = create_optimized_vggnet()

# Compile VGGnet model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation reduce training time
datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)

datagen2.fit(x_train)

# Train VGGNet model
history2 = model2.fit(
    datagen2.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 64
)

# Evaluate VGGNet model
test_loss2, test_acc2 = model2.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc2}')
print(f'Test loss: {test_loss2}')

# Plot VGGNet training and validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot VGGNet training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history2.history['accuracy'], label='Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# SECTION 3: Problem 3 ResNet-18 Model
# Define the ResNet-18 Model
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10): # change number of classes based on dataset used
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(model, trainloader, testloader, optimizer, criterion, num_epochs=20):
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(trainloader))
        val_losses.append(val_loss / len(testloader))
        train_acc.append(100 * correct / total)
        val_acc.append(100 * correct_val / total_val)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss / len(trainloader):.4f}, "
              f"Train Accuracy: {100 * correct / total:.2f}%, "
              f"Val Loss: {val_loss / len(testloader):.4f}, "
              f"Val Accuracy: {100 * correct_val / total_val:.2f}%")

    return train_losses, val_losses, train_acc, val_acc


if __name__ == '__main__':
    # Data loading and model training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Uncomment for Cifar10
    #trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform) # Uncomment fir Cifar100
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # Uncomment fir Cifar10
    #testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform) # Uncomment fir Cifar100
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train ResNet-18 model
    train_losses, val_losses, train_acc, val_acc = train(model, trainloader, testloader, optimizer, criterion)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
