import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
# from torchsummary import summary
# from torchstat import stat
# from torch.optim.lr_scheduler import StepLR
import time

torch.manual_seed(42)

class LeNet(nn.Module):
    # def __init__(self):
    #     super(LeNet, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 4 * 4, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 4 * 4)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(14*14, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        # self.fc2 = nn.Linear(128, 10)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        # x = self.fc2(x)
        return x

script_dir = os.path.dirname(__file__)

transform = transforms.Compose([transforms.ToTensor()])


print("path= "+script_dir)
trainset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, './data'), download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, './data'), download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=48, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

val_data_iter = iter(testloader)
val_image, val_label = val_data_iter.__next__()
print(val_image.size())


model = LeNet()
model = model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss_list = []
for epoch in range(3):
    running_loss = 0.0
    # for inputs, labels in trainloader:
    print('epoch ',epoch)  


    start_time = time.time()

    count = 0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        count += 1

    end_time = time.time()
    train_time = end_time - start_time

    loss_list.append(running_loss)

    start_time = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    test_time = end_time - start_time

    print('[%d] train_loss: %.3f  test_accuracy: %.3f, train_time: %.3f, test_time: %.3f' %
          (epoch + 1, running_loss / count, correct/total, train_time, test_time))  

    # if epoch > 10 and running_loss >= loss_list[-7]:
    #     break
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(correct, correct/total)  

for name, param in model.named_parameters():
    print(f"Layer name: {name}")
    print(f"Parameter shape: {param.shape}")
    np.savetxt(os.path.join(script_dir, f'./{name}.txt'), param.detach().cpu().numpy().flatten())