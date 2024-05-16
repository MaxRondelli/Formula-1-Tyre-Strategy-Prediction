import torch
import numpy as np
import torch.nn as nn
from model import GRUModel
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    

# Reproducibility
torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

'''
STEP 1: LOADING DATASET
'''
dataset = np.load("lap_data.npy", allow_pickle=True) # Load the dataset

# Assuming the last column is the target variable and the rest are features
X = dataset[:, :-1]
y = dataset[:, -1]

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = X_train.shape[1]
hidden_dim = 64
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 1

model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

# Use GPU for model
if torch.cuda.is_available():
    model.cuda()
     
'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()
 
'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

'''
STEP 7: TRAIN THE MODEL
'''
# Number of steps to unroll
seq_dim = 1 # 1 since each row is trated as a sequence
loss_list = []
iter = 0
num_epochs = 500
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # Reshape input tensor
        features = features.view(-1, seq_dim, input_dim)

        # Use GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
          
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        outputs = model(features)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels.view(-1, 1))

        if torch.cuda.is_available():
            loss.cuda()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        
        loss_list.append(loss.item())
        iter += 1
         
        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            for features, labels in test_loader:
                features = features.view(-1, seq_dim, input_dim)
                if torch.cuda.is_available():
                    features = features.cuda()
                
                outputs = model(features)
                predicted = outputs.data > 0.5  # Assuming binary classification for accuracy calculation
                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu().view(-1) == labels.cpu()).sum()
                else:
                    correct += (predicted.view(-1) == labels).sum()
            
            accuracy = 100 * correct / total
            
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

# Evaluate the final model
model.eval()
with torch.no_grad():
    test_loss = 0
    for features, labels in test_loader:
        features = features.view(-1, seq_dim, input_dim)
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        outputs = model(features)
        loss = criterion(outputs, labels.view(-1, 1))
        test_loss += loss.item()
    test_loss /= len(test_loader)
    print('Test Loss: {}'.format(test_loss))