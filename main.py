import torch
import torch.nn as nn
import numpy as np
import time
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, cohen_kappa_score
from pandas import DataFrame
from AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec


# Define your ConvTran model
#class ConvTran(nn.Module):
#    def __init__(self, input_shape, nb_classes):
#        super(ConvTran, self).__init__()
#        self.resnet = nn.Linear(np.prod(input_shape), nb_classes)

#    def forward(self, x):
#        x = x.view(x.size(0), -1)
#        return self.resnet(x)


# Define your ConvTran model
class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape']

        #print(config['Data_shape'])

        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out

# Function to get batches
def get_batch(array, i, batch_size):
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, array.shape[0])
    batch_array = array[start_id:end_id]
    return batch_array

# Function to train the model
def train(model, x_train_S2_pixel, y_train, loss_function, optimizer, BATCH_SIZE):
    tot_loss = 0
    iterations = x_train_S2_pixel.shape[0] // BATCH_SIZE
    if x_train_S2_pixel.shape[0] % BATCH_SIZE != 0:
        iterations += 1

    for ibatch in range(iterations):
        batch_x_S2_p = get_batch(x_train_S2_pixel, ibatch, BATCH_SIZE)
        batch_y = get_batch(y_train, ibatch, BATCH_SIZE)
        
        # Convert batch_y to a PyTorch tensor
        batch_y = torch.tensor(batch_y)

        optimizer.zero_grad()
        mainEstim = model(batch_x_S2_p)
        loss = loss_function(mainEstim, batch_y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    return tot_loss / iterations


# Define the batch size and number of epochs
BATCH_SIZE = 34
n_epochs = 100



# Load data and convert to PyTorch tensors
train_x = torch.tensor(np.load('/mnt/DATA/AZZA/3_Transformation-2D/Data/splits/dordogne/train_X_pxl_1.npy'))
test_x = torch.tensor(np.load('/mnt/DATA/AZZA/3_Transformation-2D/Data/splits/dordogne/test_X_pxl_1.npy'))
valid_x = torch.tensor(np.load('/mnt/DATA/AZZA/3_Transformation-2D/Data/splits/dordogne/valid_X_pxl_1.npy'))



# Load the numpy array from the file
train_y= np.load('/mnt/DATA/AZZA/3_Transformation-2D/Data/splits/dordogne/train_y_1.npy')
train_y_tensor = torch.from_numpy(train_y)
print("Shape of train_y_tensor:", train_y_tensor.shape)

test_y= np.load('/mnt/DATA/AZZA/3_Transformation-2D/Data/splits/dordogne/test_y_1.npy')
test_y_tensor = torch.from_numpy(test_y)
print("Shape of test_y_tensor:", test_y_tensor.shape)

valid_y= np.load('/mnt/DATA/AZZA/3_Transformation-2D/Data/splits/dordogne/valid_y_1.npy')
valid_y_tensor = torch.from_numpy(train_y)
print("Shape of train_y_tensor:", valid_y_tensor.shape)
print("num_classes =",np.unique(train_y))


# Subtract 1 from each label to bring them into the range expected by PyTorch (0 to num_classes-1)
train_y = train_y - 1
test_y = test_y - 1
valid_y = valid_y - 1

# Convert the updated labels to PyTorch tensors
train_y_tensor = torch.from_numpy(train_y)
test_y_tensor = torch.from_numpy(test_y)
valid_y_tensor = torch.from_numpy(valid_y)


print("Unique values in train_y:", np.unique(train_y))




# Initialize the model, loss function, and optimizer
config = {'Data_shape': (23, 4), 'emb_size': 64, 'num_heads': 8, 'dim_ff': 256, 'Fix_pos_encode': 'tAPE', 'Rel_pos_encode': 'eRPE', 'dropout': 0.1}
num_classes = 7  
model = ConvTran(config, num_classes)


loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define the path for saving the model checkpoint
checkpoint_path = "/mnt/DATA/AZZA/convtran/model_checkpoint.pt"

# Initialize the best F-measure variable
best_valid_fMeasure = 0

# Train the model
for e in range(n_epochs):
    start = time.time()
    trainLoss = train(model, train_x, train_y, loss_function, optimizer, BATCH_SIZE)
    end = time.time()
    elapsed = end - start

    # Validate the model
    with torch.no_grad():
        valid_pred = model(valid_x)
        valid_pred = torch.argmax(valid_pred, dim=1)
        fscore = metrics.f1_score(valid_y, valid_pred, average="weighted")

        # Save the model if F1 score improves
        if fscore > best_valid_fMeasure:
            best_valid_fMeasure = fscore
            torch.save(model.state_dict(), checkpoint_path)
            print('Model saved')

    print(f"Epoch {e} with loss {trainLoss:.4f} and F-Measure on validation {fscore:.4f} in {elapsed:.4f} seconds")

# Load the best model
model.load_state_dict(torch.load(checkpoint_path))
print('Model loaded')

# Evaluate the model on the test set
with torch.no_grad():
    test_pred = model(test_x)
    test_pred = torch.argmax(test_pred, dim=1)

accuracy = metrics.accuracy_score(test_y, test_pred)
f1_score = metrics.f1_score(test_y, test_pred, average="weighted")
kappa = metrics.cohen_kappa_score(test_y, test_pred)
conf_matrix = metrics.confusion_matrix(test_y, test_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1_score:.4f}')
print(f'Kappa: {kappa:.4f}')

