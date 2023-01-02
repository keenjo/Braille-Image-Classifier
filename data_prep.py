import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from skimage import io, transform
import fnmatch
import string
from tqdm import tqdm
from shutil import copyfile

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
        # Convert string labels to integers
        labels = []
        for path in self.image_path_list:
            label = path.replace(self.data_dir + 'sorted_data/', '')[0]
            labels.append(label)
        labels = sorted(list(set(labels)))
        labels2tensor = {label: labels.index(label) for label in labels}

        image_path_ex = self.image_path_list[index]
        label_ex = image_path_ex.replace(self.data_dir + 'sorted_data/', '')[0]
        # Load image and transform it into a tensor as a grayscale image (since the images don't contain any colors other than black, white, gray)
        image_ex = io.imread(image_path_ex)
        # Normalize image (make values between 0 and 1)
        image_ex = image_ex / np.max(image_ex)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Grayscale(num_output_channels=1)])
        # Transform image and get label's corresponding integer
        image_ex = transform(image_ex)
        label_ex = labels2tensor[label_ex]

        return image_ex, label_ex

    def prep_data(self, data_dir):
        '''
        Function to organize the letter images into one directory for each letter
        - this function assumes the dataset was downloaded and unzipped in the same directory as the python scripts
        '''
        os.makedirs(f'{data_dir}/sorted_data/', exist_ok=True)  # Creates a sorted_data directory within Braille Dataset directory
        for root, dirs, files in os.walk(f'{data_dir}/Braille Dataset'):
            for file in tqdm(sorted(files)):
                if file.endswith('.jpg'):
                    os.makedirs(f'{data_dir}/sorted_data/{file[0]}/',
                                exist_ok=True)  # Adds a directory for each letter of the alphabet
                    copyfile(f'{root}/{file}',
                             f'{data_dir}/sorted_data/{file[0]}/{file}')  # Adds each letter image to it's corresponding directory

    def _find_files(self, directory):
        '''
        Function to get all files in data directory
        '''
        image_path_list = []
        sorted_dir = os.path.join(directory, "sorted_data")
        if not os.path.isdir(sorted_dir):
            print("Processing the data.")
            self.prep_data(self.data_dir)
        for letter in string.ascii_lowercase:
            curr_dir = os.path.join(sorted_dir, letter)
            image_path_list += [os.path.join(curr_dir, f) for f in os.listdir(curr_dir)]
        return image_path_list


'''
In practice we should probably just import the ImageDataset class into another file 
so we can organize our data (using the code below) in the same file where the model will be built
- The only reason I initially added the code below in this file was to make sure all of the functions 
  above were working correctly with the DataLoader

dataset = ImageDataset()
print(f'{len(dataset)} images in the dataset')

# Splitting data into train, test, val splits
# 1248, 156, 156 correspond to 80%, 10%, and 10% of the dataset respectively
split_data = random_split(dataset, [1248, 156, 156], generator=torch.Generator().manual_seed(54))
train_data, val_data, test_data = split_data

# Example of a training image
train_image_ex, train_label_ex = train_data[0]
print(f'Image shape: {train_image_ex.shape}')
print(f'Label: {train_label_ex}')
print(train_image_ex)

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
'''