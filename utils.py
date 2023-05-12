import os
import random
import shutil


def split_total_images_folder(DATA_PATH):
    """Create two folders for train and validation set

    Args:
        DATA_PATH (str): the data path
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
        split_ratio = 0.99

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
