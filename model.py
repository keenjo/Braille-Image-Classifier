import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io, transform
import fnmatch


data_dir = '.../Braille_Dataset/sorted_data/'


class ImageDataset(Dataset):

    def __init__(self, data_dir):
        # Initialization of data directory and list of all of the paths to each image in the data
        self.data_dir = data_dir
        self.image_path_list = sorted(self._find_files(data_dir))

    def __len__(self):
        '''
        Function to get the length of the dataset
        '''
        return len(self.image_path_list)

    def __getitem__(self, index):
        '''
        Function to be able to select images and corresponding labels from the dataset
        '''
        image_path_ex = self.image_path_list[index]
        label_ex = image_path_ex.replace(self.data_dir, '')[0]
        image_ex = io.imread(image_path_ex)
        image_ex = torch.tensor(image_ex, dtype=float)

        return image_ex, label_ex

    def _find_files(self, directory, pattern='*.jpg'):
        '''
        Function to get all files in data directory
        '''
        image_path_list = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                image_path_list.append(os.path.join(root, filename))
        return image_path_list

# Testing ImageDataset and Dataloader classes
dataset = ImageDataset(data_dir)
#print(f'{len(dataset)} images in the dataset')

image_ex, label_ex = dataset[0]
#print(f'Image shape: {image_ex.shape}')

batch_size = 5
#print(f'Batch size: {batch_size}')
image_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

image_batch_data, image_batch_name = next(iter(image_dataloader))
#print(f'Batch shape [batch_size, image_shape]: {image_batch_data.shape}')
#print('Number of batches:', len(image_dataloader))