import torch
from torch import nn
from torch.autograd import Variable

from torchvision import datasets, transforms

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP= 28
INPUT_SIZE=28
LR = 0.01

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    100, shuffle=True)



class RNNMINST(nn.module):
    def __init__(self):
        pass



