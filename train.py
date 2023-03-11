import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import dataset # importing dataset class from data.py
from models import ConvNet # importing ConvNet class from models.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchsummary

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# hyperparameters
batch_size = 32
num_epochs = 2

# load data
composed_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150))]) # sequential transform
dataset = dataset(root_dir='Adversarial Attacks on Weather Dataset/Multi-class Weather Dataset', transform=composed_transform) # loading in dataset

# split data into training and testing set
train_size = math.ceil(0.7*dataset.__len__())
test_size = dataset.__len__() - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

def imshow(img):  # function to plot an image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# loading the data
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True) # this function automatically loads our dataset 
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

classes = ('Cloudy', 'Rain', 'Shine', 'Sunrise')

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

# loading the model
model = ConvNet().to(device=device)
torchsummary.summary(model, (3, 150, 150))

# defining the loss function and the optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

n_total_steps = len(train_loader)
print(n_total_steps)
for epoch in range(num_epochs): 
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss(outputs, labels)

        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0: # print progress
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth' # save trained model to path
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(4)] # for getting class accuracies
    n_class_samples = [0 for i in range(4)]
    for images, labels in test_loader: # predicting on the test set
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1) # get our predictions
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item() # number correct
        
        for i in range(batch_size): # 
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1 # for each class, seeing whether predictions and labels are same
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')