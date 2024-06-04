import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import dataset_statistics, joinSequences, createFileSequences, embeddingsGeneration
from dataloader import load_data

# Define your feedforward neural network
class FFNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define dataset class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define remapping function
def remap(all_unique_labels, class_label_dict):
    correspondence_list = []

    for label in all_unique_labels:
        if label in class_label_dict.values():
            corresponding_key = next(key for key, value in class_label_dict.items() if value == label)
            correspondence_list.append(corresponding_key)
        else:
            correspondence_list.append(None)

    return correspondence_list

# Load data
# Open the .pkl file in binary mode
#
    
#with open('/home/jpuentes/CLIP/embeddings_ProtBFD_350_balanced_allclasses.json', 'rb') as f:
#with open('/home/jpuentes/CLIP/embeddings_ProtBFD_12000_balanced.json', 'rb') as f:
#   dataProtBert = json.load(f)

tsv_file = 'output_all_classes.tsv'
faa_file = 'data_prot/refseq-pharokka-proteins.faa'
directory = 'results/includeLysis'

sequence_dict = joinSequences(faa_file)
#cls_emb = dataProtBert['cls']
#df_tsv, class_counts = load_data(tsv_file, faa_file)


#with open('/home/jpuentes/CLIP/complete_balanced_data_350.pkl', 'rb') as f:
with open('/home/jpuentes/CLIP/subsampled__balanced_data_9000.pkl', 'rb') as f:
    data = pickle.load(f)

colors = []
annots = []
phrogsAnots = []
for sublist in data:
    phrog = sublist[3]
    phrogsAnots.append(phrog)
    if sublist[2] == 'head and packaging':
        colors.append('Head & Packaging')
        annots.append(0)
    elif sublist[2] == 'tail':
        colors.append('Tail')
        annots.append(1)
    elif sublist[2] == 'lysis':
        colors.append('Lysis')
        annots.append(2)
    elif sublist[2]== 'DNA, RNA and nucleotide metabolism':
        colors.append('DNA, RNA and nucleotide metabolism')
        annots.append(3)
    elif sublist[2]== 'connector':
        colors.append('connector')
        annots.append(4)
    elif sublist[2]== 'integration and excision':
        colors.append('integration and excision')
        annots.append(5)     
    elif sublist[2]== 'transcription regulation':
        colors.append('transcription regulation')
        annots.append(6)
    elif sublist[2]== 'moron, auxiliary metabolic gene and host takeover':
        colors.append('moron, auxiliary metabolic gene and host takeover')
        annots.append(7)
    elif sublist[2]== 'other':
        colors.append('other')
        annots.append(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phrogsAnotsInt = [int(val) for val in phrogsAnots]
total_classes = len(np.unique(phrogsAnotsInt))
unique_labels = np.unique(phrogsAnotsInt)
breakpoint()
label_mapping = {original_label: i % total_classes for i, original_label in enumerate(unique_labels)}
remapped_labels = np.array([label_mapping[label] for label in phrogsAnotsInt])

# Convert data to PyTorch tensors
X = torch.tensor(cls_emb, dtype=torch.float32)
y = torch.tensor(remapped_labels)

# Define the number of folds
num_folds = 5

# Initialize lists to store evaluation metrics across folds
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
losses = []
# Initialize confusion matrix across folds
conf_matrices = []

# Perform 5-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True)

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Define dataset and dataloaders for training
    train_dataset_fold = MyDataset(X_train_fold, y_train_fold)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=64, shuffle=True)

    # Initialize the classifier for this fold
    model_fold = FFNClassifier(input_size=X.shape[1], hidden_size=128, output_size=len(np.unique(phrogsAnotsInt))).to(device)

    # Define loss function and optimizer for this fold
    criterion = nn.CrossEntropyLoss()
    optimizer_fold = optim.Adam(model_fold.parameters(), lr=0.001)
    running_loss_list = []
    # Train the classifier for this fold
    num_epochs = 1000
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader_fold:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_fold.zero_grad()
            outputs = model_fold(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_fold.step()
            running_loss += loss.item()
        running_loss_list.append(running_loss)
        print(f"Epoch {epoch+1}, Loss: {running_loss}")
    losses.append(running_loss_list)
    # Evaluate the classifier for this fold
    model_fold.eval()
    with torch.no_grad():
        X_val_fold, y_val_fold = X_val_fold.to(device), y_val_fold.to(device)
        outputs_fold = model_fold(X_val_fold)
        _, predicted_fold = torch.max(outputs_fold, 1)

        accuracy_fold = (predicted_fold == y_val_fold).sum().item() / len(y_val_fold)
        accuracy_list.append(accuracy_fold)

        precision_fold = precision_score(y_val_fold.cpu().numpy(), predicted_fold.cpu().numpy(), average=None)
        recall_fold = recall_score(y_val_fold.cpu().numpy(), predicted_fold.cpu().numpy(), average=None)
        f1_fold = f1_score(y_val_fold.cpu().numpy(), predicted_fold.cpu().numpy(), average=None)
        precision_list.append(np.mean(precision_fold))
        recall_list.append(np.mean(recall_fold))
        f1_list.append(np.mean(f1_fold))

        conf_matrix_fold = confusion_matrix(y_val_fold.cpu().numpy(), predicted_fold.cpu().numpy())
        conf_matrices.append(conf_matrix_fold)


# Compute mean metrics across folds
mean_accuracy = np.mean(accuracy_list)
mean_precision = np.mean(precision_list, axis=0)
mean_recall = np.mean(recall_list, axis=0)
mean_f1 = np.mean(f1_list, axis=0)

print("Mean Accuracy:", mean_accuracy, accuracy_list)
print("Mean Precision:", mean_precision, precision_list)
print("Mean Recall:", mean_recall, recall_list)
print("Mean F1-score:", mean_f1, f1_list)

# Compute mean confusion matrix across folds
#mean_conf_matrix = np.mean(conf_matrices, axis=0)

with open('losses_5_fold_BFD_complete.json', 'w') as f:
    json.dump(losses, f)
