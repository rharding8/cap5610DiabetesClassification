import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import numpy as np

# Focal Loss definition
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        device = inputs.device
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            self.alpha = self.alpha.to(device)
            alpha_t = self.alpha.gather(0, targets)
            F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Adjust the alpha parameter to give higher weights to underrepresented classes
# [No Diabetes, Pre-Diabetic, Diabetic]
alpha = torch.tensor([1, 5, 3], dtype=torch.float32)  # giving higher weight to "Pre-Diabetic" and "Diabetic"
focal_loss = FocalLoss(gamma=2.0, alpha=alpha)

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('./saved_model-v4')
model = DistilBertForSequenceClassification.from_pretrained('./saved_model-v4', num_labels=3)

# Load and prepare dataset
data_path = './data/diabetes-dataset/diabetes_012_health_indicators_BRFSS2015.csv'
df = pd.read_csv(data_path)

# Partition data
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:], df[['Diabetes_012']], test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

class DiabetesDataset(Dataset):
    def __init__(self, features_df, target_df, tokenizer, target_col):
        self.features_df = features_df
        self.labels = torch.tensor(target_df[target_col].values, dtype=torch.long)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx):
        features = self.features_df.iloc[idx]
        text = " ".join([str(val) for val in features])
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        label = self.labels[idx].item()
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

target_col = 'Diabetes_012'
train_dataset = DiabetesDataset(X_train_smote, y_train_smote, tokenizer, target_col)
val_dataset = DiabetesDataset(X_val, y_val, tokenizer, target_col)
test_dataset = DiabetesDataset(X_test, y_test, tokenizer, target_col)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results-v4',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    fp16=True  # Enable mixed precision training
)

# Custom loss function for the Trainer
def custom_loss(outputs, labels):
    return focal_loss(outputs.logits, labels)

# Define Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = custom_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset  # use validation dataset for evaluation
)

trainer.train()

trainer.save_model('./saved_model-v4')
tokenizer.save_pretrained('./saved_model-v4')

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model():
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    report = classification_report(labels, preds, target_names=['No Diabetes', 'Pre-Diabetic', 'Diabetic'])
    matrix = confusion_matrix(labels, preds)

    print(report)
    print(matrix)

if __name__ == "__main__":
    evaluate_model()
