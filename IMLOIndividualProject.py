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

#Dataset
train_image_datasets = Flowers102(root=data_dir, split='train', download=True, transform=flowers_transform)
val_image_datasets = Flowers102(root=data_dir, split='val', download=True, transform=flowers_transform)

#Dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloaders = torch.utils.data.DataLoader(val_image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
      x = F.relu(self.fc1(x.reshape(x.shape[0], -1)))
      x = self.fc2(x.reshape(x.shape[0], -1))

      return x

model = NeuralNetwork()
model.to(device)

lossfn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(epoch):
    model.train()
    running_loss = 0.0
    for index, data in enumerate(train_dataloaders):
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # Applying the loss function, backward and step
        outputs = model(inputs)
        outputs.to(device)

        loss = lossfn(outputs, labels)
        loss.to(device)
        #Deposits the loss pf each gradient w.r.t each parameter
        loss.backward()
        #adjust the parameters by the gradients collected in the backward pass
        optimizer.step()

def test_loop(dataloader, model, lossfn):
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
      for images, labels in dataloader:
          pred = model(images)
          test_loss += lossfn(pred, labels).item()
          correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  accuracy = 100*correct
  return accuracy

num_epochs = 100

for epoch in range(0, num_epochs):
    train(epoch)
    accuracy = test_loop(val_dataloaders, model, lossfn)
    # print accuracies
    print(f' Epoch: {epoch} | Accuracy on test data: {accuracy:.2f}%')

print(f'Finished Training.')
