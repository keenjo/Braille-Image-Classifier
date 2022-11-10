import os
from tqdm import tqdm
from shutil import copyfile

# assuming the dataset was downloaded and unzipped in this directory
def prep_data(data_dir='Braille Dataset'):
    os.makedirs(f'{data_dir}/sorted_data/', exist_ok=True) # Creates a sorted_data directory within Braille Dataset directory
    for root, dirs, files in os.walk(f'{data_dir}/Braille Dataset'):
        for file in tqdm(sorted(files)):
            if file.endswith('.jpg'):
                os.makedirs(f'{data_dir}/sorted_data/{file[0]}/', exist_ok=True) # Adds a directory for each letter of the alphabet
                copyfile(f'{root}/{file}', f'{data_dir}/sorted_data/{file[0]}/{file}') # Adds each letter image to it's corresponding directory
