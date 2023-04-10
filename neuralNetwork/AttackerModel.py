import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class AttackerNetwork(nn.Module):
       
    def __init__(self):
        super().__init__()
        
        self.linear_relu_sequence = nn.Sequential(
            nn.Linear(20, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 6)
        )

    def forward(self, x):
        logits = self.linear_relu_sequence(x)
        return logits

if __name__=='__main__':
    model = AttackerNetwork()
    model = model.to("cpu")
    print(model)