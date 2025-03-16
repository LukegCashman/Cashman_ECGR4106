# SECTION 0: Import Libraries and Prepare Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

# Vocabulary class to handle mapping between words and numerical indices
class Vocabulary:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.word_count = {}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1

def tokenize_and_pad(sentences, vocab):
    max_length = max(len(sentence.split(' ')) for sentence in sentences) + 2
    tokenized_sentences = []
    for sentence in sentences:
        tokens = [vocab.word2index["<SOS>"]] + [vocab.word2index[word] for word in sentence.split(' ')] + [vocab.word2index["<EOS>"]]
        padded_tokens = tokens + [vocab.word2index["<PAD>"]] * (max_length - len(tokens))
        tokenized_sentences.append(padded_tokens)
    return torch.tensor(tokenized_sentences, dtype=torch.long)

# Dataset class for English to French sentences
class EngFrDataset(Dataset):
    def __init__(self, pairs):
        self.eng_vocab = Vocabulary()
        self.fr_vocab = Vocabulary()
        self.pairs = []

        for eng, fr in pairs:
            self.eng_vocab.add_sentence(eng)
            self.fr_vocab.add_sentence(fr)
            self.pairs.append((eng, fr))

        self.eng_sentences = [pair[0] for pair in self.pairs]
        self.fr_sentences = [pair[1] for pair in self.pairs]
        
        self.eng_tokens = tokenize_and_pad(self.eng_sentences, self.eng_vocab)
        self.fr_tokens = tokenize_and_pad(self.fr_sentences, self.fr_vocab)

        self.eng_embedding = torch.nn.Embedding(self.eng_vocab.n_words, 100)
        self.fr_embedding = torch.nn.Embedding(self.fr_vocab.n_words, 100)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eng_tokens = self.eng_tokens[idx]
        fr_tokens = self.fr_tokens[idx]
        eng_emb = self.eng_embedding(eng_tokens)
        fr_emb = self.fr_embedding(fr_tokens)
        return eng_tokens, fr_tokens, eng_emb, fr_emb

# English-French dataset
# Flip pairs to translate french to english
english_to_french = [
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

dataset = EngFrDataset(english_to_french)

# Split dataset into training and validation sets (In this case model is being trained and validated on full dataset)
train_size = int(len(dataset))
val_size = len(dataset)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# SECTION 1: Problem 1 - GRU Model
SOS_token = 1
EOS_token = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set encoder to class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Set Decoder to class
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Define number of features
input_size = len(dataset.eng_vocab.word2index)
hidden_size = 256
output_size = len(dataset.fr_vocab.word2index)

encoder = Encoder(input_size=input_size, hidden_size=hidden_size).to(device)
decoder = Decoder(hidden_size=hidden_size, output_size=output_size).to(device)

# Set learning rate and SGD optimizer
learning_rate = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()

# Training loop with validation
def train_and_evaluate(encoder, decoder, train_dataloader, val_dataloader, n_epochs, criterion, encoder_optimizer, decoder_optimizer):
    best_val_accuracy = 0
    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()
        total_train_loss = 0

        # Training phase
        for input_tensor, target_tensor, _, _ in train_dataloader:
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)
            
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            total_train_loss += loss

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{n_epochs}, Training Loss: {avg_train_loss:.4f}')

        # Validation phase
        val_loss, val_accuracy = evaluate(encoder, decoder, val_dataloader, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Qualitative evaluation after training
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch + 1} - Generating translations for qualitative evaluation:")
            generate_translations(encoder, decoder, dataset.eng_sentences[:5])

# Training function
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Evaluation function
def evaluate(encoder, decoder, dataloader, criterion):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    correct_predictions = 0
    total_words = 0

    with torch.no_grad():
        for input_tensor, target_tensor, _, _ in dataloader:
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            # Encoding phase
            for ei in range(input_length):
                _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

            # Decoding phase
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_words = []
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                predicted_words.append(decoder_input.item())

                if decoder_input.item() == EOS_token:
                    break

            # Calculate loss
            loss = criterion(decoder_output, target_tensor[di].unsqueeze(0))
            total_loss += loss.item()

            # Calculate accuracy
            correct_predictions += sum([1 for p, t in zip(predicted_words, target_tensor.tolist()) if p == t])
            total_words += len(target_tensor)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_words

    return avg_loss, accuracy

# Generate translations for qualitative validation
def generate_translations(encoder, decoder, sentences, max_length=12):
    encoder.eval()
    decoder.eval()
    
    for sentence in sentences:
        input_tensor = torch.tensor([dataset.eng_vocab.word2index[word] for word in sentence.split(' ')]).unsqueeze(0).to(device)
        input_length = input_tensor.size(1)

        encoder_hidden = encoder.initHidden()

        # Encoding
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[0][ei], encoder_hidden)

        # Decoding with the generated output
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_token:
                break
            else:
                decoded_words.append(dataset.fr_vocab.index2word[decoder_input.item()])

        print(f"Input: {sentence}")
        print(f"Predicted translation: {' '.join(decoded_words)}\n")

# Perform training and evaluation with validation
train_and_evaluate(encoder, decoder, train_dataloader, val_dataloader, 50, criterion, encoder_optimizer, decoder_optimizer)

# SECTION 2: Problem 2 - Attention Gru Model
# Use the French vocabulary for output tokens
char_to_index = dataset.fr_vocab.word2index
index_to_char = dataset.fr_vocab.index2word

# Special tokens for the start and end of sequences
max_length = 12
SOS_token = char_to_index["<SOS>"]
EOS_token = char_to_index["<EOS>"]

# Creating a DataLoader to batch and shuffle the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Setting the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Set encoder to class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Set Decoder with attention to class
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=12, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)

        output = torch.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Set feature length
input_size = len(char_to_index)
hidden_size = 12
output_size = len(char_to_index)

encoder = Encoder(input_size=len(char_to_index), hidden_size=256).to(device)
decoder = AttnDecoder(hidden_size=256, output_size=len(char_to_index)).to(device)

# Set the learning rate and SGD optimizer
learning_rate = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

# Define training function
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=12):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Encode each character in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[char_to_index["<SOS>"]]], device=device)

    # Initial decoder hidden state is encoder's last hidden state
    decoder_hidden = encoder_hidden

    # Decoder with attention
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == char_to_index['<EOS>']:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Negative Log Likelihood Loss function for calculating loss
criterion = nn.NLLLoss()

# Set number of epochs for training
n_epochs = 41

# Training loop
for epoch in range(n_epochs):
    total_loss = 0
    for input_tensor, target_tensor, input_emb, target_emb in dataloader:
        # Move tensors to the correct device
        input_tensor = input_tensor[0].to(device)
        target_tensor = target_tensor[0].to(device)
        
        # Perform a single training step and update total loss
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        total_loss += loss
    
    # Print loss every 10 epochs
    if epoch % 10 == 0:
       print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')

# Define accuracy calculation and qualitative evaluation
def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, n_examples=5):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    correct_predictions = 0
    total_sentences = len(dataloader)

    with torch.no_grad():
        for i, (input_tensor, target_tensor, _, _) in enumerate(dataloader):
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            loss = 0

            # Encode input
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # Decode with attention
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            total_loss += loss.item() / target_length

            # Calculate accuracy: Check if predicted indices match the target tensor
            if predicted_indices == target_tensor.tolist():
                correct_predictions += 1

            if i < n_examples:
                output_sentence = ' '.join([index_to_char[idx] for idx in predicted_indices])
                target_sentence = ' '.join([index_to_char[idx.item()] for idx in target_tensor])
                print(f"Example {i+1}:")
                print(f"Input (English): {' '.join([index_to_char[idx.item()] for idx in input_tensor])}")
                print(f"Predicted (French): {output_sentence}")
                print(f"True (French): {target_sentence}\n")

    average_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_sentences) * 100
    print(f'Evaluation Loss: {average_loss}')
    print(f'Accuracy: {accuracy:.2f}%')

# Call evaluation function after training
evaluate_and_show_examples(encoder, decoder, dataloader, criterion)

# Function to perform qualitative validation
def generate_french_translation(encoder, decoder, input_sentence, max_length=12):
    encoder.eval()
    decoder.eval()

    # Tokenize and pad the input sentence
    input_tensor = torch.tensor([dataset.eng_vocab.word2index[word] for word in input_sentence.split(' ')], dtype=torch.long).unsqueeze(0).to(device)
    input_length = input_tensor.size(1)

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # Encode the input sentence
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[0][ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Prepare the initial input for the decoder (SOS token)
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    predicted_indices = []

    # Decode the sentence and generate the French translation
    for di in range(max_length):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        predicted_indices.append(topi.item())
        decoder_input = topi.squeeze().detach()

        if decoder_input.item() == EOS_token:
            break

    # Convert predicted indices to words
    predicted_sentence = ' '.join([index_to_char[idx] for idx in predicted_indices])
    return predicted_sentence

# Example for qualitative validation
# Replace with French sentances for French to English
test_sentences = [
    "I am cold",
    "We love music",
    "He enjoys reading books",
    "They travel around the world",
    "She teaches English at school"
]

print("Qualitative Validation: Generating French translations for test sentences:")
for sentence in test_sentences:
    french_translation = generate_french_translation(encoder, decoder, sentence)
    print(f"English: {sentence}")
    print(f"Predicted French: {french_translation}\n")