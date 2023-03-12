import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    An AlexNet-like model has been implemented.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 11, 4) # relu
        self.norm1 = nn.BatchNorm2d(128) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 5, padding='same') # relu
        self.norm2 = nn.BatchNorm2d(256)  
        self.pool2 = nn.MaxPool2d(3, 3) 
        self.conv3 = nn.Conv2d(256, 256, 3, 1, padding='same') # relu
        self.norm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 1, 1, padding='same') # relu
        self.norm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 1, 1, padding='same') # relu
        self.norm5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(1024, 1024) # relu
        self.dropout1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(1024, 1024) # relu
        self.dropout2 = nn.Dropout(0.5)
        self.lin3 = nn.Linear(1024, 4) # softmax

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        x = F.relu(self.conv5(x))
        x = self.norm5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        x = self.dropout1(x)
        x = F.relu(self.lin2(x))
        x = self.dropout2(x)
        x = F.softmax(self.lin3(x), dim=1)
        return x
