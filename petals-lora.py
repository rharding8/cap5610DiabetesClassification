import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

model_name = "petals-team/StableBeluga2"
INITIAL_PEERS = [
    "/ip4/127.0.0.1/tcp/40069/p2p/12D3KooWEDcW5LEfxHtsCz62YLBuhGP55f2RamHGYu3HAJu2ci4Q",
    "/ip4/127.0.0.1/tcp/33561/p2p/12D3KooWM5gg53JuwSCmXtG7NrMJged23tkzc7XLsZRWAXqL1hCD",
]

CUDA_VISIBLE_DEVICES = 0

model = AutoDistributedModelForCausalLM.from_pretrained(model_name, initial_peers=INITIAL_PEERS)
model = model.cuda()
df = pd.read_csv("diabetes_012.csv")
  
prompts = pd.read_csv("prompts.csv").head(200)
y = df['Diabetes_012'].head(200).values

tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenizePrompt(prompt):
    print(prompt)
    input = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input

tokenizePrompts = np.vectorize(tokenizePrompt)
inputs = tokenizePrompts(prompts)
class LLMBasedClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.distributed_layers = model.transformer.h
        self.adapter = nn.Sequential(nn.Linear(model.config.hidden_size, 32), nn.Linear(32, model.config.hidden_size))
        self.head = nn.Linear(model.config.hidden_size, 2)

    def forward(self, embeddings):
        mid_block = len(self.distributed_layers) // 2
        hidden_states = self.distributed_layers[:mid_block](embeddings)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.distributed_layers[mid_block:](hidden_states)
        pooled_states = torch.mean(hidden_states, dim=1)
        return self.head(pooled_states)
    
    
classifier = LLMBasedClassifier(model).cuda()
opt = torch.optim.Adam(classifier.parameters(), 3e-5)

inputs = torch.tensor(inputs, device='cuda')
labels = torch.tensor(y, device='cuda')
for i in range(10):
    loss = F.cross_entropy(classifier(inputs), labels)
    print(f"loss[{i}] = {loss.item():.3f}")
    opt.zero_grad()
    loss.backward()
    opt.step()

pred = classifier(inputs).argmax(-1)

numDiabetic = count = np.sum(pred == 2.0)
numPrediabetic = count = np.sum(pred == 1.0)
numNotDiabetic = count = np.sum(pred == 0.0)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

# Example counts for each class
true_counts = {
    "No Diabetes": np.sum(y == 0.0),
    "Pre-Diabetic": np.sum(y == 1.0),
    "Diabetic": np.sum(y == 2.0)
}

# Example predictions as percentages of support
predicted_counts = {
    "No Diabetes": numNotDiabetic,
    "Pre-Diabetic": numPrediabetic,
    "Diabetic": numDiabetic
}

# Generating example ground truth and predicted arrays
y_true = (["No Diabetes"] * true_counts["No Diabetes"] +
          ["Pre-Diabetic"] * true_counts["Pre-Diabetic"] +
          ["Diabetic"] * true_counts["Diabetic"])

y_pred = (["No Diabetes"] * int(predicted_counts["No Diabetes"]) +
          ["Pre-Diabetic"] * int(predicted_counts["Pre-Diabetic"]) +
          ["Diabetic"] * int(predicted_counts["Diabetic"]))

# Padding predictions to match the length of true labels for this example
y_pred += ["No Diabetes"] * (len(y_true) - len(y_pred))

# Calculating metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["No Diabetes", "Pre-Diabetic", "Diabetic"], zero_division=0))

# Accuracy calculation
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")