import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import pickle
import os

# -----------------------
# Config
# -----------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_CSV = r"D:\TapanProgramming\finalYearProject\ben.csv"
MAX_LEN = 50
EMBED = 256
HIDDEN = 512
BATCH_SIZE = 16  # Smaller batch for small dataset
EPOCHS = 30
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

# -----------------------
# Simple Tokenizer Class
# -----------------------
class SimpleTokenizer:
    def __init__(self, min_freq=2):
        self.word_to_id = {}
        self.id_to_word = {}
        self.min_freq = min_freq
        
    def build_vocab(self, texts):
        # Count word frequencies
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)
        
        # Build vocabulary with special tokens
        self.word_to_id = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID, 
            SOS_TOKEN: SOS_ID,
            EOS_TOKEN: EOS_ID
        }
        
        # Add words that appear at least min_freq times
        vocab_id = 4
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                self.word_to_id[word] = vocab_id
                vocab_id += 1
                
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        print(f"Vocabulary size: {len(self.word_to_id)}")
        
    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(word, UNK_ID) for word in words]
    
    def decode(self, ids):
        words = [self.id_to_word.get(id, UNK_TOKEN) for id in ids]
        # Remove special tokens and join
        words = [w for w in words if w not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
        return ' '.join(words)
    
    def get_vocab_size(self):
        return len(self.word_to_id)

# -----------------------
# Load and preprocess data
# -----------------------
print("Loading dataset...")
df = pd.read_csv(DATA_CSV, usecols=["English", "Bengali"]).dropna()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Basic cleaning - remove extra whitespace
    text = ' '.join(text.split())
    return text

df["English"] = df["English"].apply(clean_text)
df["Bengali"] = df["Bengali"].apply(clean_text)

# Filter out very short sentences
df = df[(df["English"].str.len() > 3) & (df["Bengali"].str.len() > 3)]
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Dataset size: {len(df)} pairs")

# Build tokenizers
print("Building vocabularies...")
en_tokenizer = SimpleTokenizer(min_freq=2)
bn_tokenizer = SimpleTokenizer(min_freq=2)

en_tokenizer.build_vocab(df["English"].tolist())
bn_tokenizer.build_vocab(df["Bengali"].tolist())

# -----------------------
# Dataset
# -----------------------
class SimpleTranslationDataset(Dataset):
    def __init__(self, df, en_tokenizer, bn_tokenizer, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.en_tokenizer = en_tokenizer
        self.bn_tokenizer = bn_tokenizer
        self.max_len = max_len

    def encode_sequence(self, text, tokenizer):
        ids = tokenizer.encode(text)
        ids = [SOS_ID] + ids + [EOS_ID]
        
        if len(ids) < self.max_len:
            ids += [PAD_ID] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len-1] + [EOS_ID]
        
        return ids

    def __getitem__(self, i):
        row = self.df.iloc[i]
        src = torch.tensor(self.encode_sequence(row["English"], self.en_tokenizer), dtype=torch.long)
        tgt = torch.tensor(self.encode_sequence(row["Bengali"], self.bn_tokenizer), dtype=torch.long)
        return src, tgt

    def __len__(self):
        return len(self.df)

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

train_dataset = SimpleTranslationDataset(train_df, en_tokenizer, bn_tokenizer)
val_dataset = SimpleTranslationDataset(val_df, en_tokenizer, bn_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------
# Model
# -----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_ID)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        attention = attention.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(attention, dim=1)
        
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs)
        return context.squeeze(1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_ID)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        
    def forward_step(self, input, hidden, encoder_outputs, mask):
        embedded = self.embedding(input).unsqueeze(1)
        
        context = self.attention(hidden[0], encoder_outputs, mask)
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        output, hidden = self.gru(gru_input, hidden)
        prediction = self.out(output.squeeze(1))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        vocab_size = self.decoder.out.out_features
        
        outputs = torch.zeros(batch_size, max_len-1, vocab_size).to(src.device)
        encoder_outputs, hidden = self.encoder(src)
        
        # Create mask
        mask = (src != PAD_ID).float()
        
        input = tgt[:, 0]
        
        for t in range(1, max_len):
            output, hidden = self.decoder.forward_step(input, hidden, encoder_outputs, mask)
            outputs[:, t-1] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
            
        return outputs

# Initialize model
EN_VOCAB_SIZE = en_tokenizer.get_vocab_size()
BN_VOCAB_SIZE = bn_tokenizer.get_vocab_size()

encoder = Encoder(EN_VOCAB_SIZE, EMBED, HIDDEN).to(DEVICE)
decoder = Decoder(BN_VOCAB_SIZE, EMBED, HIDDEN).to(DEVICE)
model = Seq2Seq(encoder, decoder).to(DEVICE)

print(f"English vocab size: {EN_VOCAB_SIZE}")
print(f"Bengali vocab size: {BN_VOCAB_SIZE}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -----------------------
# Training
# -----------------------
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, 0)  # No teacher forcing during eval
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

print("Starting training...")
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Gradually reduce teacher forcing
        teacher_forcing_ratio = max(0.8 - (epoch - 1) * 0.02, 0.3)
        output = model(src, tgt, teacher_forcing_ratio)
        
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    val_loss = evaluate(model, val_loader)
    avg_train_loss = total_loss / len(train_loader)
    
    print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'en_tokenizer': en_tokenizer,
            'bn_tokenizer': bn_tokenizer,
            'config': {
                'EN_VOCAB_SIZE': EN_VOCAB_SIZE,
                'BN_VOCAB_SIZE': BN_VOCAB_SIZE,
                'EMBED': EMBED,
                'HIDDEN': HIDDEN,
                'MAX_LEN': MAX_LEN
            }
        }, 'best_simple_model.pth')
        print("✓ Best model saved!")

# -----------------------
# Translation function
# -----------------------
def translate(text, model, en_tokenizer, bn_tokenizer, max_len=MAX_LEN):
    model.eval()
    
    # Clean input
    text = clean_text(text)
    if not text:
        return "Empty input"
    
    # Encode
    src_ids = en_tokenizer.encode(text)
    src_ids = [SOS_ID] + src_ids + [EOS_ID]
    
    if len(src_ids) < max_len:
        src_ids += [PAD_ID] * (max_len - len(src_ids))
    else:
        src_ids = src_ids[:max_len-1] + [EOS_ID]
    
    src = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        mask = (src != PAD_ID).float()
        
        decoded = []
        input_token = torch.tensor([SOS_ID], device=DEVICE)
        
        for _ in range(max_len):
            output, hidden = model.decoder.forward_step(input_token, hidden, encoder_outputs, mask)
            predicted = output.argmax(1)
            token_id = predicted.item()
            
            if token_id == EOS_ID:
                break
            if token_id != PAD_ID:
                decoded.append(token_id)
                
            input_token = predicted
        
        # Decode to text
        if decoded:
            translation = bn_tokenizer.decode(decoded)
            return translation if translation.strip() else "Translation failed"
        else:
            return "No translation generated"

# -----------------------
# Test translations
# -----------------------
print("\n" + "="*50)
print("🔹 TRANSLATION TESTING")
print("="*50)

test_sentences = [
    "I love you",
    "How are you?",
    "What is your name?",
    "Thank you",
    "Good morning",
    "He reads a book every day",
    "She is beautiful",
    "I am happy"
]

model.eval()
for sentence in test_sentences:
    translation = translate(sentence, model, en_tokenizer, bn_tokenizer)
    print(f"English:  {sentence}")
    print(f"Bengali:  {translation}")
    print("-" * 40)

print("\nTraining completed!")
print("Model saved as 'best_simple_model.pth'")

# Save tokenizers separately
with open('tokenizers.pkl', 'wb') as f:
    pickle.dump({
        'en_tokenizer': en_tokenizer,
        'bn_tokenizer': bn_tokenizer
    }, f)

print("Tokenizers saved as 'tokenizers.pkl'")