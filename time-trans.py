import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ====== CONFIGURATION ======
data_dir = "data"
seq_len = 128
stride = 128
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== STEP 1: LOAD DATA ======
all_chunks = []
all_labels = []
all_raw_labels = []

# First pass: collect all labels
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, file))
        if not {'heart_rate', 'breath_rate', 'label'}.issubset(df.columns):
            continue
        all_raw_labels.append(df['label'].iloc[0])

# Encode all labels once
label_encoder = LabelEncoder()
label_encoder.fit(all_raw_labels)

# Second pass: process sequences
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, file))
        if not {'heart_rate', 'breath_rate', 'label'}.issubset(df.columns):
            continue
        hr = df['heart_rate'].values.astype(np.float32)
        br = df['breath_rate'].values.astype(np.float32)
        label = df['label'].iloc[0]
        encoded_label = label_encoder.transform([label])[0]

        # Combine HR and BR into multivariate signal: shape (seq_len, 2)
        sequence = np.stack([hr, br], axis=1)  # shape: (len, 2)

        # Chunking
        for i in range(0, len(sequence) - seq_len + 1, stride):
            chunk = sequence[i:i+seq_len]  # shape: (seq_len, 2)
            all_chunks.append(chunk)
            all_labels.append(encoded_label)

print(f"Loaded {len(all_chunks)} sequences from {len(os.listdir(data_dir))} files.")
print("Classes:", list(label_encoder.classes_))

# ====== STEP 2: DATASET ======
class HRBreathDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)  # (seq_len, 2)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

X_train, X_test, y_train, y_test = train_test_split(
    all_chunks, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

train_dataset = HRBreathDataset(X_train, y_train)
test_dataset = HRBreathDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ====== STEP 3: MODEL ======
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=64, nhead=4, num_layers=2):
        super(TimeSeriesTransformer, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x) + self.pos_embedding  # (batch, seq_len, d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.fc(x)

model = TimeSeriesTransformer(
    input_dim=2,
    num_classes=len(label_encoder.classes_),
    seq_len=seq_len
).to(device)

# ====== STEP 4: TRAINING ======
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# ====== STEP 5: EVALUATION ======
from sklearn.metrics import classification_report, accuracy_score
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        output = model(x)                        # Raw logits
        probs = torch.softmax(output, dim=1)     # Convert to probabilities
        pred = torch.argmax(probs, dim=1)        # Get predicted class index

        y_true.extend(y.cpu().numpy())           # Ground truth (numeric)
        y_pred.extend(pred.cpu().numpy())        # Predictions (numeric)
        y_prob.extend(probs.max(dim=1).values.cpu().numpy())  # Max prob per sample

# Decode class names
true_labels = label_encoder.inverse_transform(y_true)
pred_labels = label_encoder.inverse_transform(y_pred)

# Print predictions with probabilities
print("\nSample Predictions with Confidence:\n")
for i in range(min(10, len(pred_labels))):  # Show first 10
    print(f"True: {true_labels[i]:<12} | Predicted: {pred_labels[i]:<12} | Confidence: {y_prob[i]*100:.2f}%")

# Accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n✅ Overall Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels))

#use belowe code for Valence ands Arousal values per video file saved in csv as 
#timestamp, valence value, arousal value , label

"""

# ====== CONFIGURATION ======
data_dir = "data"       # Folder with CSV files
seq_len = 128
stride = 128
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== STEP 1: LOAD DATA ======
all_chunks = []
all_labels = []
all_raw_labels = []

# First pass: collect labels
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, file))
        if not {'valence', 'arousal', 'label'}.issubset(df.columns):
            continue
        all_raw_labels.append(df['label'].iloc[0])

# Encode all labels
label_encoder = LabelEncoder()
label_encoder.fit(all_raw_labels)

# Second pass: process sequences
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, file))
        if not {'valence', 'arousal', 'label'}.issubset(df.columns):
            continue
        val = df['valence'].values.astype(np.float32)
        aro = df['arousal'].values.astype(np.float32)
        label = df['label'].iloc[0]
        encoded_label = label_encoder.transform([label])[0]

        # Combine valence + arousal → multivariate time series
        sequence = np.stack([val, aro], axis=1)

        for i in range(0, len(sequence) - seq_len + 1, stride):
            chunk = sequence[i:i+seq_len]
            all_chunks.append(chunk)
            all_labels.append(encoded_label)

print(f"Loaded {len(all_chunks)} chunks from {len(os.listdir(data_dir))} files.")
print("Classes:", list(label_encoder.classes_))

# ====== STEP 2: DATASET ======
class ValAroDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)  # (seq_len, 2)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

X_train, X_test, y_train, y_test = train_test_split(
    all_chunks, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

train_dataset = ValAroDataset(X_train, y_train)
test_dataset = ValAroDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ====== STEP 3: MODEL ======
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=64, nhead=4, num_layers=2):
        super(TimeSeriesTransformer, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding  # (batch, seq_len, d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

model = TimeSeriesTransformer(
    input_dim=2,
    num_classes=len(label_encoder.classes_),
    seq_len=seq_len
).to(device)

# ====== STEP 4: TRAINING ======
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# ====== STEP 5: EVALUATION ======
model.eval()
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(probs.max(dim=1).values.cpu().numpy())

true_labels = label_encoder.inverse_transform(y_true)
pred_labels = label_encoder.inverse_transform(y_pred)

print("\nSample Predictions with Confidence:\n")
for i in range(min(10, len(pred_labels))):
    print(f"True: {true_labels[i]:<12} | Predicted: {pred_labels[i]:<12} | Confidence: {y_prob[i]*100:.2f}%")

accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n✅ Overall Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels))
"""