import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader

# Ensure device is set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example English-French sentence pairs
data = [
    ("hello how are you", "bonjour comment vas tu"),
    ("what is your name", "quel est ton nom"),
    ("nice to meet you", "enchanté de vous rencontrer"),
    ("where are you from", "d'où viens-tu"),
    ("i love machine learning", "j'adore l'apprentissage automatique")
]

# Tokenization function
def tokenize(sentence):
    return sentence.lower().split()

# Build vocabulary
def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence))
    vocab = {word: idx + 3 for idx, (word, _) in enumerate(counter.most_common())}
    vocab["<pad>"] = 0
    vocab["<sos>"] = 1
    vocab["<eos>"] = 2
    return vocab

src_vocab = build_vocab([pair[0] for pair in data])
tgt_vocab = build_vocab([pair[1] for pair in data])

# Convert sentences to tensors
def encode(sentence, vocab):
    return [vocab["<sos>"]] + [vocab.get(token, vocab["<pad>"]) for token in tokenize(sentence)] + [vocab["<eos>"]]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = [torch.tensor(encode(s, src_vocab), device=device) for s in src_batch]
    tgt_batch = [torch.tensor(encode(t, tgt_vocab), device=device) for t in tgt_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_vocab["<pad>"])
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab["<pad>"])
    return src_padded, tgt_padded

dataloader = DataLoader(data, batch_size=2, collate_fn=collate_fn, shuffle=True)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, device=device).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim).to(device)
        self.positional_encoding = PositionalEncoding(model_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True).to(device)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(device)
        self.fc_out = nn.Linear(model_dim, output_dim).to(device)

    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(src.shape[-1])
        tgt = self.embedding(tgt) * math.sqrt(tgt.shape[-1])
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return self.fc_out(output)

# Define model parameters
input_dim = len(src_vocab)
output_dim = len(tgt_vocab)
model_dim = 64
num_heads = 4
num_layers = 2

model = SimpleTransformer(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"]).to(device)

# Training loop
model.train()
for epoch in range(5):
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        optimizer.zero_grad()
        prediction = model(src, tgt_input)
        prediction = prediction.view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)
        loss = criterion(prediction, tgt_output)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete!")

# Save model
torch.save(model.state_dict(), "transformer_model.pth")
print("Model saved!")

# Load model
def load_model():
    model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
    model.eval()
    print("Model loaded!")

# Inference function
def translate(sentence):
    model.eval()
    src_tensor = torch.tensor(encode(sentence, src_vocab), device=device).unsqueeze(0)
    tgt_tensor = torch.tensor([tgt_vocab["<sos>"]], device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(20):  # Limit output length
            output = model(src_tensor, tgt_tensor)
            next_token = output.argmax(dim=-1)[:, -1].item()
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == tgt_vocab["<eos>"]:
                break
    translation = [word for word, idx in tgt_vocab.items() if idx in tgt_tensor.squeeze().tolist()]
    return " ".join(translation[1:])

# Example usage
load_model()
sentence = "hello how are you"
translation = translate(sentence)
print(f"Input: {sentence}\nTranslation: {translation}")
