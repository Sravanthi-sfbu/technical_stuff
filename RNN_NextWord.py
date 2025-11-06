import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.utils import to_categorical
import torch.nn.functional as F
import numpy as np

# 1.INPUT TEXT DATA

text = "SFBU focuses on innovative learning "
text = text*2

text_len = len(text)
print("Text length:", text_len)

# Unique characters
chars = sorted(list(set(text)))
print("Unique characters:", chars)

# Create mappings
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)
print("Vocabulary size:", vocab_size)

# Create character sequences (3 chars -> next char)
seq_length = 3
sequences, labels = [], []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[c] for c in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

# One-hot encode
X_one_hot = to_categorical(X, num_classes=vocab_size)
y_one_hot = to_categorical(y, num_classes=vocab_size)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_one_hot, dtype=torch.float32)
y_tensor = torch.tensor(y_one_hot, dtype=torch.float32)

print("Input shape:", X_tensor.shape)
print("Output shape:", y_tensor.shape)


# 2.MODEL: LSTM (Better memory than Simple RNN)

class CharLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CharLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step output
        out = self.fc(out)
        return out

# Initialize model
input_dim = vocab_size
hidden_dim = 128  # larger hidden size for better sequence learning
output_dim = vocab_size

model = CharLSTM(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3.TRAINING (200 EPOCHS)

epochs = 200
print("\nTraining started...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, torch.argmax(y_tensor, dim=1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == torch.argmax(y_tensor, dim=1)).float().mean()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - Accuracy: {acc.item():.4f}")

print("Training complete!\n")

# 4.GENERATE NEW TEXT (with Temperature Sampling)

def sample_with_temperature(logits, temperature=0.9):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()

start_seq = "SFB"   # starting seed
generated_text = start_seq

for i in range(100):  # generate 100 new characters
    x_input = np.array([[char_to_index[c] for c in generated_text[-seq_length:]]])
    x_input = to_categorical(x_input, num_classes=vocab_size)
    x_tensor = torch.tensor(x_input, dtype=torch.float32)

    with torch.no_grad():
        output = model(x_tensor)
        temperature = 0.7   # try 0.3, 0.7, 0.9, 1.0
        next_index = sample_with_temperature(output[0], temperature)

        next_char = index_to_char[next_index]
        generated_text += next_char

print("Generated Text:")
print(generated_text)
