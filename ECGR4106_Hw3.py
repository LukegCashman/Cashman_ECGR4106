# SECTION 0: Library and dataset prep
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import requests

# Sample text
text = """Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.
At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.
One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.
Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.
Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.
In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."""

chars = sorted(list(set(text)))
ix_to_char = {i: ch for i, ch in enumerate(chars)}
char_to_ix = {ch: i for i, ch in enumerate(chars)} 
chars = sorted(list(set(text)))

# Preparing the dataset
max_length = 30
X = []
y = []
for i in range(len(text) - max_length):
    sequence = text[i:i + max_length]
    label = text[i + max_length]
    X.append([char_to_ix[char] for char in sequence])
    y.append(char_to_ix[label])

X = np.array(X)
y = np.array(y)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Download the Shakespear dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

# Preparing the dataset
sequence_length = 30
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integers
encoded_text = [char_to_int[ch] for ch in text]

# Create sequences and targets
sequences = []
targets = []
for i in range(0, len(encoded_text) - sequence_length):
    seq = encoded_text[i:i+sequence_length]
    target = encoded_text[i+sequence_length]
    sequences.append(seq)
    targets.append(target)

# Convert lists to PyTorch tensors
sequences = torch.tensor(sequences, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

# Create a dataset class
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

# Instantiate the dataset
dataset = CharDataset(sequences, targets)

# Create data loaders
batch_size = 128
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Use multiple workers for loading data
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True)

# SECTION 1: Problem 1 - RNN Model
# Defining the RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.005
epochs = 100

# Model, loss, and optimizer
model = CharRNN(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        _, predicted = torch.max(val_output, 1)
        val_accuracy = (predicted == y_val).float().mean()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

# Predicting the next character
test_str = "Next character prediction is a fundamental task in the field of nat"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")

# SECTION 2: Problem 1 - LSTM Model
# Defining the LSTM model
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hn, cn) = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.005
epochs = 100

# Model, loss, and optimizer
model = CharLSTM(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        _, predicted = torch.max(val_output, 1)
        val_accuracy = (predicted == y_val).float().mean()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

# Predicting the next character
test_str = "Next character prediction is a fundamental task in the field of nat"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")

# SECTION 3: Problem 1 - Gru Model
# Defining the GRU model
class CharGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hn = self.gru(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.005
epochs = 100

# Model, loss, and optimizer
model = CharGRU(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        _, predicted = torch.max(val_output, 1)
        val_accuracy = (predicted == y_val).float().mean()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

# Predicting the next character
test_str = "Next character prediction is a fundamental task in the field of nat"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")

# SECTION 4: Problem 2 - Basic LSTM & Gru Model
# Defining the LSTM model
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hn, cn) = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.005
epochs = 50

# Model, loss, and optimizer
model = CharLSTM(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if GPU is available and move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mixed Precision Training
scaler = torch.amp.GradScaler('cuda')

if __name__ == '__main__':
    # Training the model
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        for batch_idx, (sequences_batch, targets_batch) in enumerate(train_loader):
            sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
            
            optimizer.zero_grad()

            # Mixed Precision Forward and Backward Pass
            with torch.amp.autocast('cuda'):
                output = model(sequences_batch)
                loss = criterion(output, targets_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for sequences_batch, targets_batch in test_loader:
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
                
                val_output = model(sequences_batch)
                loss = criterion(val_output, targets_batch)
                val_loss += loss.item()

                _, predicted = torch.max(val_output, 1)
                val_accuracy += (predicted == targets_batch).float().mean().item()

        # Average loss and accuracy for validation
        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)

        # Print statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Defining the GRU model
class CharGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hn = self.gru(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.005
epochs = 50

# Model, loss, and optimizer
model = CharGRU(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if GPU is available and move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mixed Precision Training
scaler = torch.amp.GradScaler('cuda')

if __name__ == '__main__':
    # Training the model
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        for batch_idx, (sequences_batch, targets_batch) in enumerate(train_loader):
            sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
            
            optimizer.zero_grad()

            # Mixed Precision Forward and Backward Pass
            with torch.amp.autocast('cuda'):
                output = model(sequences_batch)
                loss = criterion(output, targets_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for sequences_batch, targets_batch in test_loader:
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
                
                val_output = model(sequences_batch)
                loss = criterion(val_output, targets_batch)
                val_loss += loss.item()

                _, predicted = torch.max(val_output, 1)
                val_accuracy += (predicted == targets_batch).float().mean().item()

        # Average loss and accuracy for validation
        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)

        # Print statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# SECTION 5: Problem 2 - Complex LSTM & Gru Model
# Defining the LSTM model with multiple hidden layers and larger hidden size
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hn, cn) = self.lstm(embedded)
        output = output[:, -1, :]
        
        # Pass through additional FC layers
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return output

# Hyperparameters
hidden_size = 256
num_layers = 3
learning_rate = 0.001
epochs = 20
dropout = 0.2

# Model, loss, and optimizer
model = CharLSTM(len(chars), hidden_size, len(chars), num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if GPU is available and move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mixed Precision Training
scaler = torch.amp.GradScaler('cuda')

if __name__ == '__main__':
    # Training the model
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        for batch_idx, (sequences_batch, targets_batch) in enumerate(train_loader):
            sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
            
            optimizer.zero_grad()

            # Mixed Precision Forward and Backward Pass
            with torch.amp.autocast('cuda'):
                output = model(sequences_batch)
                loss = criterion(output, targets_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for sequences_batch, targets_batch in test_loader:
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
                
                val_output = model(sequences_batch)
                loss = criterion(val_output, targets_batch)
                val_loss += loss.item()

                _, predicted = torch.max(val_output, 1)
                val_accuracy += (predicted == targets_batch).float().mean().item()

        # Average loss and accuracy for validation
        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)

        # Print statistics
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Defining the GRU model with multiple hidden layers and larger hidden size
class CharGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Adding GRU instead of LSTM
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hn = self.gru(embedded)
        output = output[:, -1, :]
        
        # Pass through additional FC layers
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return output

# Hyperparameters
hidden_size = 256
num_layers = 3
learning_rate = 0.001
epochs = 20
dropout = 0.2

# Model, loss, and optimizer
model = CharGRU(len(chars), hidden_size, len(chars), num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if GPU is available and move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mixed Precision Training
scaler = torch.amp.GradScaler('cuda')

if __name__ == '__main__':
    # Training the model
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        for batch_idx, (sequences_batch, targets_batch) in enumerate(train_loader):
            sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
            
            optimizer.zero_grad()

            # Mixed Precision Forward and Backward Pass
            with torch.amp.autocast('cuda'):
                output = model(sequences_batch)
                loss = criterion(output, targets_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for sequences_batch, targets_batch in test_loader:
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
                
                val_output = model(sequences_batch)
                loss = criterion(val_output, targets_batch)
                val_loss += loss.item()

                _, predicted = torch.max(val_output, 1)
                val_accuracy += (predicted == targets_batch).float().mean().item()

        # Average loss and accuracy for validation
        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)

        # Print statistics
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
