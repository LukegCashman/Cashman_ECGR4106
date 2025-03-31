# SECTION 0: Library intialization
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# SECTION 1: Problem 1
# Sample text
text = """At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.
            One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.
            Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.'
            'Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.
            In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."""

# Creating character vocabulary
chars = sorted(list(set(text)))
ix_to_char = {i: ch for i, ch in enumerate(chars)}
char_to_ix = {ch: i for i, ch in enumerate(chars)}

# Preparing the dataset
max_length = 10  # adjust for sequence lengths 20 and 30
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

# Defining the Transformer model
class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return output

# Hyperparameters
hidden_size = 128
num_layers = 3
nhead = 2
learning_rate = 0.005
epochs = 100

# Model, loss, and optimizer
model = CharTransformer(len(chars), hidden_size, len(chars), num_layers, nhead)
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
test_str = "This is a simple example to demonstrate how to predict the next char"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")

# SECTION 2: Problem 2
# Download the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

# Prepare the dataset
max_length = 20 # adjust for sequence lengths 30 and 50
# Create a character mapping to integers
chars = sorted(list(set(text)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integers
encoded_text = [char_to_ix[ch] for ch in text]

# Create sequences and targets
sequences = []
targets = []
for i in range(0, len(encoded_text) - max_length):
    seq = encoded_text[i:i+max_length]
    target = encoded_text[i+max_length]
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

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Defining the Transformer model
class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return output

# Hyperparameters
hidden_size = 64
num_layers = 2  # Adjust for 4 layers
nhead = 2       # Adjust for 4 heads
learning_rate = 0.005
epochs = 5

# Model, loss, and optimizer
model = CharTransformer(len(chars), hidden_size, len(chars), num_layers, nhead)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (sequences_batch, targets_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(sequences_batch)
        loss = criterion(output, targets_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences_batch, targets_batch in test_loader:
            val_output = model(sequences_batch)
            val_loss = criterion(val_output, targets_batch)
            total_val_loss += val_loss.item()

            _, predicted = torch.max(val_output, 1)
            correct += (predicted == targets_batch).sum().item()
            total += targets_batch.size(0)

    avg_loss = total_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(test_loader)
    val_accuracy = correct / total

    if (epoch+1) % 1 == 0:
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

# Predicting the next character
test_str = "This is a simple example to demonstrate how to predict the next char"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")

# SECTION 3: Problem 3
# Sample dataset of English-French
# Flip pairs to translate French to English
text = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    ("She is happy", "Elle est heureuse"),
    ("We are friends", "Nous sommes amis"),
    ("They are students", "Ils sont étudiants"),
    ("The cat is sleeping", "Le chat dort"),
    ("The sun is shining", "Le soleil brille"),
    ("We love music", "Nous aimons la musique"),
    ("She speaks French fluently", "Elle parle français couramment"),
    ("He enjoys reading books", "Il aime lire des livres"),
    ("They play soccer every weekend", "Ils jouent au football chaque week-end"),
    ("The movie starts at 7 PM", "Le film commence à 19 heures"),
    ("She wears a red dress", "Elle porte une robe rouge"),
    ("We cook dinner together", "Nous cuisinons le dîner ensemble"),
    ("He drives a blue car", "Il conduit une voiture bleue"),
    ("They visit museums often", "Ils visitent souvent des musées"),
    ("The restaurant serves delicious food", "Le restaurant sert une délicieuse cuisine"),
    ("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
    ("We watch movies on Fridays", "Nous regardons des films le vendredi"),
    ("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
    ("They travel around the world", "Ils voyagent autour du monde"),
    ("The book is on the table", "Le livre est sur la table"),
    ("She dances gracefully", "Elle danse avec grâce"),
    ("We celebrate birthdays with cake", "Nous célébrons les anniversaires avec un gâteau"),
    ("He works hard every day", "Il travaille dur tous les jours"),
    ("They speak different languages", "Ils parlent différentes langues"),
    ("The flowers bloom in spring", "Les fleurs fleurissent au printemps"),
    ("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
    ("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
    ("The dog barks loudly", "Le chien aboie bruyamment"),
    ("He sings beautifully", "Il chante magnifiquement"),
    ("They swim in the pool", "Ils nagent dans la piscine"),
    ("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
    ("She teaches English at school", "Elle enseigne l'anglais à l'école"),
    ("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
    ("He paints landscapes", "Il peint des paysages"),
    ("They laugh at the joke", "Ils rient de la blague"),
    ("The clock ticks loudly", "L'horloge tic-tac bruyamment"),
    ("She runs in the park", "Elle court dans le parc"),
    ("We travel by train", "Nous voyageons en train"),
    ("He writes a letter", "Il écrit une lettre"),
    ("They read books at the library", "Ils lisent des livres à la bibliothèque"),
    ("The baby cries", "Le bébé pleure"),
    ("She studies hard for exams", "Elle étudie dur pour les examens"),
    ("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
    ("He fixes the car", "Il répare la voiture"),
    ("They drink coffee in the morning", "Ils boivent du café le matin"),
    ("The sun sets in the evening", "Le soleil se couche le soir"),
    ("She dances at the party", "Elle danse à la fête"),
    ("We play music at the concert", "Nous jouons de la musique au concert"),
    ("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
    ("They study French grammar", "Ils étudient la grammaire française"),
    ("The rain falls gently", "La pluie tombe doucement"),
    ("She sings a song", "Elle chante une chanson"),
    ("We watch a movie together", "Nous regardons un film ensemble"),
    ("He sleeps deeply", "Il dort profondément"),
    ("They travel to Paris", "Ils voyagent à Paris"),
    ("The children play in the park", "Les enfants jouent dans le parc"),
    ("She walks along the beach", "Elle se promène le long de la plage"),
    ("We talk on the phone", "Nous parlons au téléphone"),
    ("He waits for the bus", "Il attend le bus"),
    ("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
    ("The stars twinkle at night", "Les étoiles scintillent la nuit"),
    ("She dreams of flying", "Elle rêve de voler"),
    ("We work in the office", "Nous travaillons au bureau"),
    ("He studies history", "Il étudie l'histoire"),
    ("They listen to the radio", "Ils écoutent la radio"),
    ("The wind blows gently", "Le vent souffle doucement"),
    ("She swims in the ocean", "Elle nage dans l'océan"),
    ("We dance at the wedding", "Nous dansons au mariage"),
    ("He climbs the mountain", "Il gravit la montagne"),
    ("They hike in the forest", "Ils font de la randonnée dans la forêt"),
    ("The cat meows loudly", "Le chat miaule bruyamment"),
    ("She paints a picture", "Elle peint un tableau"),
    ("We build a sandcastle", "Nous construisons un château de sable"),
    ("He sings in the choir", "Il chante dans le chœur"),
    ("They ride bicycles", "Ils font du vélo"),
    ("The coffee is hot", "Le café est chaud"),
    ("She wears glasses", "Elle porte des lunettes"),
    ("We visit our grandparents", "Nous rendons visite à nos grands-parents"),
    ("He plays the guitar", "Il joue de la guitare"),
    ("They go shopping", "Ils font du shopping"),
    ("The teacher explains the lesson", "Le professeur explique la leçon"),
    ("She takes the train to work", "Elle prend le train pour aller au travail"),
    ("We bake cookies", "Nous faisons des biscuits"),
    ("He washes his hands", "Il se lave les mains"),
    ("They enjoy the sunset", "Ils apprécient le coucher du soleil"),
    ("The river flows calmly", "La rivière coule calmement"),
    ("She feeds the cat", "Elle nourrit le chat"),
    ("We visit the museum", "Nous visitons le musée"),
    ("He fixes his bicycle", "Il répare son vélo"),
    ("They paint the walls", "Ils peignent les murs"),
    ("The baby sleeps peacefully", "Le bébé dort paisiblement"),
    ("She ties her shoelaces", "Elle attache ses lacets"),
    ("We climb the stairs", "Nous montons les escaliers"),
    ("He shaves in the morning", "Il se rase le matin"),
    ("They set the table", "Ils mettent la table"),
    ("The airplane takes off", "L'avion décolle"),
    ("She waters the plants", "Elle arrose les plantes"),
    ("We practice yoga", "Nous pratiquons le yoga"),
    ("He turns off the light", "Il éteint la lumière"),
    ("They play video games", "Ils jouent aux jeux vidéo"),
    ("The soup smells delicious", "La soupe sent délicieusement bon"),
    ("She locks the door", "Elle ferme la porte à clé"),
    ("We enjoy a picnic", "Nous profitons d'un pique-nique"),
    ("He checks his email", "Il vérifie ses emails"),
    ("They go to the gym", "Ils vont à la salle de sport"),
    ("The moon shines brightly", "La lune brille intensément"),
    ("She catches the bus", "Elle attrape le bus"),
    ("We greet our neighbors", "Nous saluons nos voisins"),
    ("He combs his hair", "Il se peigne les cheveux"),
    ("They wave goodbye", "Ils font un signe d'adieu")
]

# Creating word vocabulary for both English and French
en_vocab = set()
fr_vocab = set()

for en, fr in text:
    en_vocab.update(en.split())
    fr_vocab.update(fr.split())

en_vocab = sorted(list(en_vocab))
fr_vocab = sorted(list(fr_vocab))

en_vocab_size = len(en_vocab)
fr_vocab_size = len(fr_vocab)

# Mapping words to indices
en_to_ix = {word: i for i, word in enumerate(en_vocab)}
ix_to_en = {i: word for i, word in enumerate(en_vocab)}
fr_to_ix = {word: i for i, word in enumerate(fr_vocab)}
ix_to_fr = {i: word for i, word in enumerate(fr_vocab)}

# Padding token
PAD_IDX = 0

# Prepare the dataset
class TranslationDataset(Dataset):
    def __init__(self, data, en_to_ix, fr_to_ix, max_len=10):
        self.data = data
        self.en_to_ix = en_to_ix
        self.fr_to_ix = fr_to_ix
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        en_sentence, fr_sentence = self.data[idx]
        en_indices = [self.en_to_ix[word] for word in en_sentence.split()]
        fr_indices = [self.fr_to_ix[word] for word in fr_sentence.split()]
        
        # Padding sequences
        en_indices = en_indices + [PAD_IDX] * (self.max_len - len(en_indices)) if len(en_indices) < self.max_len else en_indices[:self.max_len]
        fr_indices = fr_indices + [PAD_IDX] * (self.max_len - len(fr_indices)) if len(fr_indices) < self.max_len else fr_indices[:self.max_len]
        
        return torch.tensor(en_indices), torch.tensor(fr_indices)

# Create the dataset
dataset = TranslationDataset(text, en_to_ix, fr_to_ix)

# Splitting the dataset into training and validation sets
train_data, val_data = train_test_split(dataset.data, test_size=0.2, random_state=42)

train_dataset = TranslationDataset(train_data, en_to_ix, fr_to_ix)
val_dataset = TranslationDataset(val_data, en_to_ix, fr_to_ix)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Define the Transformer model for Seq2Seq (Encoder-Decoder)
class Seq2SeqTransformer(nn.Module):
    def __init__(self, en_vocab_size, fr_vocab_size, hidden_size, num_layers, nhead):
        super(Seq2SeqTransformer, self).__init__()
        
        self.en_embedding = nn.Embedding(en_vocab_size, hidden_size)
        self.fr_embedding = nn.Embedding(fr_vocab_size, hidden_size)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead),
            num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead),
            num_layers
        )
        
        self.fc_out = nn.Linear(hidden_size, fr_vocab_size)
    
    def forward(self, src, tgt):
        # Embed the source and target sequences
        src_emb = self.en_embedding(src)
        tgt_emb = self.fr_embedding(tgt)
        
        # Pass through the encoder
        memory = self.encoder(src_emb)
        
        # Pass through the decoder
        output = self.decoder(tgt_emb, memory)
        
        # Output layer to get the predicted word logits
        output = self.fc_out(output)
        return output

# Hyperparameters
hidden_size = 128
num_layers = 3      # Adjust to 4 to increase complexity
nhead = 2           # Adjust to 4 to increase complexity
learning_rate = 0.005
epochs = 100

# Model, loss, and optimizer
model = Seq2SeqTransformer(en_vocab_size, fr_vocab_size, hidden_size, num_layers, nhead)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for en_seq, fr_seq in train_loader:
        optimizer.zero_grad()
        
        # Reshape inputs to be (sequence_length, batch_size, feature_size)
        en_seq = en_seq.T  # Shape: (seq_len, batch_size)
        fr_seq_input = fr_seq[:, :-1].T  # Exclude the last token of the target sequence
        fr_seq_target = fr_seq[:, 1:].T  # Shifted target sequence for loss calculation
        
        # Forward pass
        output = model(en_seq, fr_seq_input)  # Exclude the last token of the target sequence
        loss = criterion(output.reshape(-1, fr_vocab_size), fr_seq_target.reshape(-1))  # Shift target
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for en_seq, fr_seq in val_loader:
            en_seq = en_seq.T  # Shape: (seq_len, batch_size)
            fr_seq_input = fr_seq[:, :-1].T  # Exclude the last token of the target sequence
            fr_seq_target = fr_seq[:, 1:].T  # Shifted target sequence for loss calculation
            
            output = model(en_seq, fr_seq_input)
            loss = criterion(output.reshape(-1, fr_vocab_size), fr_seq_target.reshape(-1))
            val_loss += loss.item()

            # Get the predicted indices
            _, predicted = torch.max(output, dim=-1)  # Shape: (seq_len, batch_size)
            
            # Calculate accuracy by comparing the predicted tokens to the actual target tokens
            correct_predictions += (predicted == fr_seq_target).sum().item()
            total_predictions += fr_seq_target.numel()

    val_accuracy = 100 * correct_predictions / total_predictions  # Convert to percentage

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Translation function
def translate(model, sentence, en_to_ix, ix_to_fr, max_len=10):
    model.eval()
    tokens = sentence.split()
    token_indices = [en_to_ix.get(word, PAD_IDX) for word in tokens]
    token_indices = torch.tensor(token_indices).unsqueeze(0)  # Adding batch dimension (1, seq_len)
    
    # Initialize the target sequence (e.g., <sos> token)
    tgt = torch.zeros((1, max_len), dtype=torch.long)
    
    # Perform the translation
    with torch.no_grad():
        for i in range(1, max_len):  # Start from 1 because the first token is <sos>
            output = model(token_indices.T, tgt[:, :i].T)  # Note: output is (batch_size, seq_len, vocab_size)
            
            # Ensure the sequence length does not exceed model output
            seq_len = output.size(1)
            if i >= seq_len:
                break  # Stop if we reach the end of the model's output sequence
            
            # Get the most probable token for the current timestep (i)
            next_word_idx = torch.argmax(output[0, i, :]).item()  # Select from the batch index (0)
            tgt[0, i] = next_word_idx
            
            if next_word_idx == PAD_IDX:  # Stop if we hit padding token
                break
    
    # Translate indices to words, excluding padding
    translated_words = [ix_to_fr[idx.item()] for idx in tgt[0] if idx != PAD_IDX]
    return " ".join(translated_words)

# Testing translation
test_sentence = "I am cold" # Use french translation for poroblem 4
translated_sentence = translate(model, test_sentence, en_to_ix, ix_to_fr)
print(f"Original: {test_sentence}")
print(f"Translated: {translated_sentence}")