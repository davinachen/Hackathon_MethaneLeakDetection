import os
import random
import shutil
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets


def split_total_images_folder(DATA_PATH, split_ratio=0.8):
    """Create two folders for train and validation set

    Args:
        DATA_PATH (str): the data path
        split_ratio (float between 0 and 1): the split ratio between train and val
    """

    # Define the paths
    original_folder = f'{DATA_PATH}/images'
    train_folder = f'{DATA_PATH}/train_images'
    val_folder = f'{DATA_PATH}/val_images'

    # Check if the split has already been done
    if os.path.exists(train_folder) and os.path.exists(val_folder):
        print("Split has already been done. Exiting...")

    else:
        # Create the train and validation folders
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # Define the split ratio (80% for training, 20% for validation)
        

        # Walk through the original folder and copy images to train and validation folders
        for root, dirs, files in os.walk(original_folder):
            for subdir in dirs:
                original_subfolder = os.path.join(root, subdir)
                train_subfolder = os.path.join(train_folder, subdir)
                val_subfolder = os.path.join(val_folder, subdir)

                os.makedirs(train_subfolder, exist_ok=True)
                os.makedirs(val_subfolder, exist_ok=True)

                images = os.listdir(original_subfolder)
                random.shuffle(images)

                split_index = int(len(images) * split_ratio)
                train_images = images[:split_index]
                val_images = images[split_index:]

                for image in train_images:
                    src = os.path.join(original_subfolder, image)
                    dst = os.path.join(train_subfolder, image)
                    shutil.copyfile(src, dst)

                for image in val_images:
                    src = os.path.join(original_subfolder, image)
                    dst = os.path.join(val_subfolder, image)
                    shutil.copyfile(src, dst)

        print("Split completed successfully.")



def dataset_creation(image_transforms, DATA_PATH, bs=16):
    """create the datasets objects

    Args:
        image_transforms (transforms): the image transformations to do
        DATA_PATH (str): the data path
        bs (int, optional): the batch_size. Defaults to 3.


    Returns:
        data, train_data, valid_data, train_data_size, valid_data_size

    """

    # Load the Data
    # Set train and valid directory paths
    train_directory = f'{DATA_PATH}/train_images'
    valid_directory = f'{DATA_PATH}/val_images'


    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    }

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    # Create iterators for the Data loaded using DataLoader module
    train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)

    return data, train_data, valid_data, train_data_size, valid_data_size



def plot_images(data, labels_map, cols=3, rows=3):
    """plot some images

    Args:
        data (dict): dictionnary of the train and validation dataset
        labels_map (dict): the dictionnary linking the integer and plume/no_plume
        cols (int, optional): Number of cols. Defaults to 3.
        rows (int, optional): Number of rows. Defaults to 3.
    """
    figure = plt.figure(figsize=(8, 8))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data['train']), size=(1,)).item()
        img, label = data['train'][sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()