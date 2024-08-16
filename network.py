import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets, transforms # Import transforms
from torchvision.transforms import ToTensor, Resize

# Get + load data
transformation = transforms.Compose([
      transforms.ToTensor()

    ])

train = datasets.MNIST(
    root="data",
    download = True,
    transform = transformation
    )
dataset = DataLoader(train, 32)


# Core CLF NN
class ConvNet(nn.Module):
  def __init__ (self):
    super().__init__()

    self.initial_model = nn.Sequential (
        #this is def. overkill

        #layer 1
        nn.Conv2d(1,32,(3,3)),
        nn.ReLU(),

        #layer 2
        nn.Conv2d(32,64,(3,3)),
        nn.ReLU(),

        #layer 3
        nn.Conv2d(64,64,(3,3)),
        nn.ReLU(),

        #layer 4
        nn.Conv2d(64,128,(3,3)),
        nn.ReLU(),


        nn.Flatten(),
        nn.Linear(128*(28-8)*(28-8), 10)

    )
  def forward(self,x):
    return self.initial_model(x)