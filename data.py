import os
from torch.utils.data import Dataset
from skimage import io
from skimage.color import rgba2rgb
import pandas as pd
import numpy as np

class dataset(Dataset):
    """
    This class allows Pytorch to load our dataset into the DataLoader library, according to Pytorch, in order to make a custom dataset,
    2 functions need to be defined: 
    __len__(): This returns the length of the dataset 
    __getitem__(idx): This returns a single item of the dataset at a specified index idx
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir # the root directory where our images are located
        self.transform = transform # how we want to transform our images
        
    def __len__(self):
        count=0
        for folder in os.listdir(self.root_dir): # for each class within the root directory
            folder_path = os.path.join(self.root_dir, folder) 
            count += len(os.listdir(folder_path)) # total number of images = number of images within each subfolder
        return count
    
    def __getitem__(self, idx):
        idx_list = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            idx_list += [(img, folder) for img in os.listdir(folder_path)] # append the file names of each sub-folder to the list
        image_path = os.path.join(self.root_dir, idx_list[idx][1]) # root directory/class path
        image_path = os.path.join(image_path, idx_list[idx][0]) # root directory/class/image id path
        image = io.imread(image_path) # read in the image (by default this should be rgb but im not sure what's exactly going on)
        try:
            image = rgba2rgb(image) # hard-converting the image to rgb, if there are more/less dimensions
        except:
            pass 
        y_label = idx_list[idx][1]
        classes = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        y_label_df = pd.DataFrame(index=classes, data=np.eye(4))
        if self.transform:
            image = self.transform(image)
        return (image, y_label_df.loc[y_label].to_numpy()) # each y_label we return should be of the form [0, 0, ... 1(where label is true), ..., 0]