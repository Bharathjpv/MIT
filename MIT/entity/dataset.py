import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MitDataset(Dataset):
    def __init__(self, annotation_file_path, 
                 image_dir, 
                 class_to_index,
                 transformation=None
                 ):
        self.data_frame = pd.read_csv(annotation_file_path, names=['path'])
        self.image_dir = image_dir
        self.transformation = transformation
        self.class_to_index = class_to_index
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):

    
        label = self._get_image_sample_label(idx)
        label =  self.class_to_index[label]
        label = torch.tensor(label)

        image_sample_path = self._get_image_sample_path(idx)
        img = Image.open(image_sample_path).convert('RGB')

        if self.transformation:
            image = self.transformation(img)
        
        return image, label
    
    def _get_image_sample_path(self, idx):
        
        relative_path = self.data_frame['path'][idx]
        path = os.path.join(self.image_dir, relative_path)
        return path

    def _get_image_sample_label(self, idx):

        relative_path = self.data_frame['path'][idx]
        return relative_path.split('/')[0]
