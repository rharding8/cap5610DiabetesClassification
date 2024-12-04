import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split  # Add this line
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


# Custom Dataset class
class DiabetesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        if alpha is None:
            alpha = [1, 1, 1]
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_loss = torch.mean(alpha_t * (1 - pt) ** self.gamma * ce_loss)
        return focal_loss

# Enhanced CNN
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 4, 6)
            output_shape = self.conv3(self.conv2(self.conv1(dummy_input))).view(1, -1).shape[1]
        
        self.attention = nn.Sequential(
            nn.Linear(output_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc1 = nn.Linear(output_shape, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        attn_weights = self.attention(x)
        x = x * attn_weights
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Complex CNN
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 2), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 4, 6)
            output_shape = self.conv3(self.conv2(self.conv1(dummy_input))).view(1, -1).shape[1]
        
        self.attention = nn.Sequential(
            nn.Linear(output_shape, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc1 = nn.Linear(output_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        attn_weights = self.attention(x)
        x = x * attn_weights
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training and saving function
def train_and_save_model(model_type, hyperparams, train_loader, val_loader, test_loader, device="cpu"):
    if model_type == "basic":
        model = EnhancedCNN().to(device)
    elif model_type == "complex":
        model = ComplexCNN().to(device)
    else:
        raise ValueError("Invalid model type. Choose 'basic' or 'complex'.")

    lr = hyperparams.get("lr", 0.001)
    weight_decay = hyperparams.get("weight_decay", 1e-5)
    num_epochs = hyperparams.get("num_epochs", 50)
    patience = hyperparams.get("patience", 10)
    class_weights = hyperparams.get("class_weights", [1, 3, 1.5])
    gamma = hyperparams.get("gamma", 2)

    criterion = FocalLoss(alpha=class_weights, gamma=gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    counter = 0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history['train_losses'].append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                val_outputs = model(val_features)
                val_loss += criterion(val_outputs, val_labels).item()

        val_loss /= len(val_loader)
        history['val_losses'].append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_features, test_labels in test_loader:
            test_features, test_labels = test_features.to(device), test_labels.to(device)
            test_outputs = model(test_features)
            _, preds = torch.max(test_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    metrics = {
        "confusion_matrix": cm_percentage,
        "classification_report": classification_report(all_labels, all_preds)
    }

    print("\nConfusion Matrix (Percentages):\n", cm_percentage)
    print("\nClassification Report:\n", metrics["classification_report"])

    return model, history, metrics

# CNNCode
def CNNCode(model, X_train, y_train, X_test, y_test, X_val, y_val):
    y_train = y_train.astype(float).astype(int)
    y_val = y_val.astype(float).astype(int)
    y_test = y_test.astype(float).astype(int)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    y_train = np.array(y_train)

    # Preprocess features for CNN
    def preprocess_features(features):
        X_padded = np.zeros((features.shape[0], 24))
        X_padded[:, :features.shape[1]] = features
        return X_padded.reshape(-1, 1, 4, 6)

    X_train_tensor = torch.tensor(preprocess_features(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(preprocess_features(X_val), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)

    X_test_tensor = torch.tensor(preprocess_features(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    train_dataset = DiabetesDataset(X_train_tensor, y_train_tensor)
    val_dataset = DiabetesDataset(X_val_tensor, y_val_tensor)
    test_dataset = DiabetesDataset(X_test_tensor, y_test_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams = {
        "lr": 0.0005,
        "weight_decay": 1e-5,
        "num_epochs": 50,
        "patience": 10,
        "class_weights": [1, 3, 1.5],
        "gamma": 2
    }
    if model == 1:
        print("\nTraining Complex Model")
        return train_and_save_model(
            model_type="complex", 
            hyperparams=hyperparams, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader, 
            device=device
        )
    if model == 2:
        print("Training Basic Model")
        return train_and_save_model(
            model_type="basic", 
            hyperparams=hyperparams, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader, 
            device=device
        )
