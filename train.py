import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import dataset # importing dataset class from data.py
from models import ConvNet, pretrainedConvNet  # importing ConvNet class from models.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchsummary
from attacks import create_attacked_training_set

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/MCWD")

def train(model, preprocess = None, PATH = None, dataset = dataset, dataset_dir = "Multi-class Weather Dataset"):
    """Completes our training process for a model, where data is preprocessed using a transform preprocess,
    and sends path to path. By default if preprocess is none no preprocessing is performed. If path = None
    then the data is not saved."""
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparameters
    batch_size = 32
    num_epochs = 10

    # load data
    trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) # transform to make greyscale images have the same channels
    composed_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150)), trans]) # sequential transform
    dataset = dataset(root_dir=dataset_dir, transform=composed_transform) # loading in dataset

    # split data into training and testing set
    train_size = math.ceil(0.7*dataset.__len__())
    test_size = dataset.__len__() - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator = generator1)

    def imshow(img):  # function to plot an image
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()


    # loading the data
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True) # this function automatically loads our dataset
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    classes = ('Cloudy', 'Rain', 'Shine', 'Sunrise')

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # add images to tensorboard
    imgs = torchvision.utils.make_grid(images)
    writer.add_image('example_images', imgs)

    # loading the model
    model = model.to(device=device)
    torchsummary.summary(model, (3, 150, 150))

    # defining the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # saving model graph to tensorboard
    writer.add_graph(model, images.to(device, dtype=torch.float))

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0
        running_correct = 0
        n_samples = 0
        print("Running Epoch",epoch)
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images = images.to(device, dtype=torch.float)
            if(preprocess):
                images = preprocess(images)
            labels = labels.to(device, dtype=torch.float)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            # reshape labels
            labels = torch.argmax(labels, dim = 1)
            n_samples += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if (i+1) % 5 == 0: # print progress
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        writer.add_scalar(f'training_loss_every_epoch', running_loss/100, epoch)
        writer.add_scalar(f'training_accuracy_every_epoch', 100*running_correct/n_samples, epoch)

    writer.close()

    print('Finished Training')
    print(f'Training accuracy of the network: {100*running_correct/n_samples} %')
    if(PATH):
        torch.save(model.state_dict(), PATH)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)] # for getting class accuracies
        n_class_samples = [0 for i in range(4)]

        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            if preprocess:
                images = preprocess(images)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(images)

            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1) # get our predictions

            # Reshape Labels
            labels = torch.argmax(labels, dim = 1)

            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item() # number correct

            for i in range(len(labels)):
               label = labels[i]
               pred = predicted[i]
               if label == pred:
                   n_class_correct[label] += 1 # for each class, seeing whether predictions and labels are same
               n_class_samples[label] += 1

        test_acc = 100.0 * n_correct / n_samples
        print(f'Testing accuracy of the network: {test_acc} %')

        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
    return model

model = ConvNet()
model.load_state_dict(torch.load('cnn.pth'))
trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) # transform to make greyscale images have the same channels
composed_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150)), trans]) # sequential transform
create_attacked_training_set(model, 'Multi-class Weather Dataset', composed_transform)
train(ConvNet(), PATH = './cnn_attacked.pth', dataset_dir = 'MCWD_attacked')
#train(pretrainedConvNet(), preprocess = transforms.Compose([
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
#    PATH = './alex-cnn_attacked.pth')