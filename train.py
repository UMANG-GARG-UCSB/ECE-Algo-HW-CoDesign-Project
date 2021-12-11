import numpy as np
import torch
import itertools
import scipy
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

from pydub import AudioSegment

# datapath = 'C:\/Users\/Yu\/Desktop\/train\/'
datapath = 'D:\/Study_Files\/UCSB\/Courses\/ECE 594BB Hardware for AI\/ProjectWorkSpace\/train\/'

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(
    precision=4,
    sci_mode=False
)

np.set_printoptions(suppress=True)


# Reproducity;
np.random.seed(1137)
torch.manual_seed(114514)

# Load data;
stft_data_max_unroll = np.load(datapath + 'stftdata.npy')

print(stft_data_max_unroll.dtype)

frame_size = stft_data_max_unroll.shape[2]


# Package training data and labels;
train_data = torch.zeros([3, 4, frame_size])
for i in range(3):
    for j in range(4):
        train_data[i, j] = torch.tensor(stft_data_max_unroll[i, j])
train_data = train_data.reshape(-1, frame_size)

train_label = torch.zeros([12], dtype=torch.long)
for i in range(0, 4):
    train_label[i] = torch.tensor([0])
for i in range(4, 8):
    train_label[i] = torch.tensor([1])
for i in range(8, 12):
    train_label[i] = torch.tensor([2])

# Package test data;
cat_test = torch.tensor(stft_data_max_unroll[0, 4], dtype=torch.float64)
apple_test = torch.tensor(stft_data_max_unroll[1, 4], dtype=torch.float64)
box_test = torch.tensor(stft_data_max_unroll[2, 4], dtype=torch.float64)


# Configuration of MLP model;
input_feature = frame_size
hidden_feature = 32
output_feature = 3
num_hidden_layer = 1

lr = 1e-3
epochs = 100

# Loss function: use crossentropy based loss;
l = torch.nn.CrossEntropyLoss(reduction='mean')

# MLP definition;
model = torch.nn.Sequential(
    torch.nn.Linear(input_feature, hidden_feature),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_feature, output_feature)
)

# Optimizer definition;
optim = torch.optim.SGD(model.parameters(), lr=lr)

# Main training loop;
loss_plot = np.zeros([epochs])
for epoch in tqdm(range(epochs)):
    optim.zero_grad()
    prediction = model(train_data)
    loss = l(prediction, train_label)
    loss_plot[epoch] = loss.detach().numpy()
    loss.backward()
    optim.step()

# Plot training loss;
plt.figure(13)
plt.title('Training loss')
plt.plot(loss_plot)
plt.savefig(datapath + 'plot\/' 'train_loss' + '.png')

# Validation with traininng data;
y = torch.nn.Softmax(dim=1)(model(train_data))
y_np = y.detach().numpy()
y_label = np.argmax(y_np, axis=1)
print('Validation result is:', y_label)

# Test data;
print('Start testing cases:')
y1 = torch.nn.Softmax(dim=0)(model(cat_test)).detach().numpy()
print('Cat test result is:', np.argmax(y1, axis=0))
y2 = torch.nn.Softmax(dim=0)(model(apple_test)).detach().numpy()
print('Apple test result is:', np.argmax(y2, axis=0))
y3 =torch.nn.Softmax(dim=0)(model(box_test)).detach().numpy()
print('Box test result is:', np.argmax(y3, axis=0))

# Get the model trained parameters;
w1 = model[0].weight.detach().numpy()
b1 = model[0].bias.detach().numpy()
w2 = model[2].weight.detach().numpy()
b2 = model[2].bias.detach().numpy()

np.save(datapath + 'w1.npy', w1)
np.save(datapath + 'b1.npy', b1)
np.save(datapath + 'w2.npy', w2)
np.save(datapath + 'b2.npy', b2)

def np_relu (a, dtype=np.float):
    dim = a.shape[0]
    b = np.zeros(dim, dtype=dtype)
    for i in range(dim):
        if (a[i] > 0): b[i] = a[i]
        else: b[i] = 0
    return  b

def np_crossentropy(a):
    return np.exp(a) / np.sum(np.exp(a))

print(torch.nn.Softmax(dim=0)(model(cat_test)))

c1 = np_relu(w1 @ stft_data_max_unroll[0][4] + b1)
c2 = w2 @ c1 + b2

print(np_crossentropy(c2))