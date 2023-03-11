import os
from torch.utils.data import Dataset
from skimage import io
from skimage.color import rgba2rgb
import pandas as pd
import numpy as np

class dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        count=0
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            count += len(os.listdir(folder_path))
        return count
    
    def __getitem__(self, idx):
        idx_list = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            idx_list += [(img, folder) for img in os.listdir(folder_path)] # append the file names of each sub-folder to the list
        image_path = os.path.join(self.root_dir, idx_list[idx][1])
        image_path = os.path.join(image_path, idx_list[idx][0])
        image = io.imread(image_path)
        try:
            image = rgba2rgb(image)
        except:
            pass
        y_label = idx_list[idx][1]
        classes = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        y_label_df = pd.DataFrame(index=classes, data=np.eye(4))
        if self.transform:
            image = self.transform(image)
        return (image, y_label_df.loc[y_label].to_numpy())