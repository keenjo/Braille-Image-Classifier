import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from skimage import io, transform
import fnmatch
import string
from prep_data import prep_data

class ImageDataset(Dataset):

    def __init__(self, data_dir="Braille Dataset/"):
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
        # Transform images into tensors
        # !! Could experiment with normalizing the image tensors as well
        image_ex = torch.tensor(image_ex, dtype=float)

        return image_ex, label_ex

    def _find_files(self, directory):
        '''
        Function to get all files in data directory
        '''
        image_path_list = []
        sorted_dir = os.path.join(directory, "sorted_data")
        if not os.path.isdir(sorted_dir):
            print("Processing the data.")
            prep_data(self.data_dir)
        for letter in string.ascii_lowercase:
            curr_dir = os.path.join(sorted_dir, letter)
            image_path_list += [os.path.join(curr_dir, f) for f in os.listdir(curr_dir)]
        return image_path_list


# Testing ImageDataset and Dataloader classes
dataset = ImageDataset()
print(f'{len(dataset)} images in the dataset')

# Splitting data into train, test, split
# 1248, 156, 156 correspond to 80%, 10%, and 10% of the dataset respectively
split_data = random_split(dataset, [1248, 156, 156], generator=torch.Generator().manual_seed(54))
train_data = split_data[0]
test_data = split_data[1]
val_data = split_data[2]

# Example of a training image
train_image_ex, train_label_ex = train_data[0]
print(f'Image shape: {train_image_ex.shape}')

# Creating the train, test, and val dataloaders
batch_size = 5
print(f'Batch size: {batch_size}')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# Looking at an example in the training batch
train_batch_data, train_batch_name = next(iter(train_dataloader))
print(f'Batch shape [batch_size, image_shape]: {train_batch_data.shape}')
print('Number of batches:', len(train_dataloader))
