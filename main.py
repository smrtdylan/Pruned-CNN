from network import ConvNet 
from torch import nn
from torch.optim import Adam
clf = ConvNet().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()