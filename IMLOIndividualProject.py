import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import Flowers102

batch_size = 4
data_dir = '... /... /Data/flowers-102'

flowers_transform =  transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_image_datasets = Flowers102(root=data_dir, split='train', download=True, transform=flowers_transform)
val_image_datasets = Flowers102(root=data_dir, split='val', download=True, transform=flowers_transform)

train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloaders = torch.utils.data.DataLoader(val_image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device: ",device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(64*64*8, 512)
        self.fc2 = nn.Linear(512, 102)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = self.fc1(x.reshape(x.shape[0], -1))
      x = self.fc2(x.reshape(x.shape[0], -1))

      return x

model = NeuralNetwork()
model.to(device)
