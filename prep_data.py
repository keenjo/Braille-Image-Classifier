import os
from shutil import copyfile

dir = '.../Braille_Dataset' # Enter the directory to your Braille Dataset
os.makedirs(f'{dir}/sorted_data/', exist_ok=True) # Creates a sorted_data directory within Braille Dataset directory

for root, dirs, files in os.walk(dir):
    for file in sorted(files):
        if file.endswith('.jpg'):
            print(os.path.join(root, file))
            os.makedirs(f'{dir}/sorted_data/{file[0]}/', exist_ok=True) # Adds a directory for each letter of the alphabet
            copyfile(f'{root}/{file}', f'{dir}/sorted_data/{file[0]}/{file}') # Adds each letter image to it's corresponding directory