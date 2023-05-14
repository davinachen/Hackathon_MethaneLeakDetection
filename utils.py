import os
import random
import shutil
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score, roc_auc_score
import streamlit as st


# metadata cleaning
def preprocess_metadata(DATA_PATH, read=True):
    """process the metadata

    Args:
        metadata (pandas dataframe): the metadata
    """
    if read:
        metadata = pd.read_pickle(f'{DATA_PATH}/metadata_processed.pkl')
    else:
        pd.read_csv(f'{DATA_PATH}/metadata.csv')
        metadata['date'] = pd.to_datetime(metadata['date'], format='%Y%m%d')
        metadata = metadata.drop(columns=['set'])

        # Create an instance of the geocoder
        geolocator = Nominatim(user_agent="my-app")

        # Use reverse geocoding to get the location information
        metadata['location'] = metadata.progress_apply(lambda x: geolocator.reverse((x.lat, x.lon)), axis=1)

        # Extract the country from the location information
        metadata['country'] = metadata.location.progress_apply(lambda x: x.raw['address'].get('country'))
        metadata['city'] = metadata.location.progress_apply(lambda x: x.raw['address'].get('city'))
        
        #save as pkl 
        metadata.to_pickle(f'{DATA_PATH}/metadata_processed.pkl')

    return metadata



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




def train(DATA_PATH, model, epochs, loss_criterion, optimizer, train_data, val_data, device, model_name, scheduler=None, performances_df=None):
    """train a given model to perform the task

    Args:
        model (pytorch model): the model to train
        epochs (int): number of epochs to train
        loss_criterion (torch.nn): the loss on which perform the training
        optimizer (dataset): the torch optimizer 
        train_data (dataset): the training data
        val_data (dataset): the validation data
        device (str): the device to perform the calculation ('cpu', or 'cuda')
        model_name (str): the name to give to the model_saving file
        scheduler (pandas df, optional): the learning rate scheduler. Defaults to None.
        performances_df (pandas df, optional): If different from None, continue the completion of the given datast. Defaults to None.

    Returns:
        performances_df: the summary of the training
    """
    
    if performances_df == None:
        if scheduler != None:
            performances_df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'val_loss', 'train_accu', 'val_accu', 'train_auc', 'val_auc'])
        else:
            performances_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_accu', 'val_accu', 'train_auc', 'val_auc'])
    
    best_loss = np.inf
    
    for epoch in range(epochs):
        epoch_start = time.time()
        # print("Epoch: {}/{}".format(epoch+1, epochs))
        # Set to training mode
        model.train()
        # Loss and Accuracy within the epoch
        train_loss = []
        train_accu = []
        train_auc = []

        val_loss = []
        val_accu = []
        val_auc = []

        for i, (inputs, labels) in enumerate(train_data):
            ################
            ###  TRAIN   ###
            ################
            inputs = inputs.to(device)
            labels = labels.to(device)
            numpy_labels = labels.cpu().bool().numpy()
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            ## LOSS ##

            # Compute loss
            loss = loss_criterion(outputs, labels)
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            #scheduler
            try:
                scheduler.step() #update the lambda scheduler
            except:
                pass
            # Compute the total loss for the batch and add it to train_loss
            train_loss.append(loss.item())

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            train_accu.append(accuracy_score(numpy_labels,predictions.cpu().bool().numpy()))
            
            #compute auc_roc
            probas_plume = outputs[:,-1:].detach().cpu().numpy()
            try:
                train_auc.append(roc_auc_score(numpy_labels, probas_plume))
            except:
                print('Only one class in y_true !!')
                train_auc.append(0)

            ################
            ###  VALID   ###
            ################
            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_data):

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    numpy_labels = labels.cpu().bool().numpy()
                    outputs = model(inputs)

                    # Compute loss
                    loss = loss_criterion(outputs, labels)
                    val_loss.append(loss.item())

                    # Compute the accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    val_accu.append(accuracy_score(numpy_labels, predictions.cpu().bool().numpy()))
                    
                    #compute auc_roc
                    probas_plume = outputs[:,-1:].detach().cpu().numpy()
                    try:
                        val_auc.append(roc_auc_score(numpy_labels, probas_plume))
                    except:
                        print('Only one class in y_true !!')
                        val_auc.append(0)

        val_loss = np.mean(val_loss)
        try:
            lr = scheduler.get_last_lr()[0]
            for_df_list = [
                        epoch, lr, np.mean(train_loss), val_loss, 
                        np.mean(train_accu), np.mean(val_accu),
                        np.mean(train_auc), np.mean(val_auc)
                        ]
        except:
            lr = scheduler.get_last_lr()[0]
            for_df_list = [
                        epoch, np.mean(train_loss), val_loss, 
                        np.mean(train_accu), np.mean(val_accu),
                        np.mean(train_auc), np.mean(val_auc)
                        ]
        
        performances_df.loc[len(performances_df)] =  for_df_list                                 

        if epoch%10 == 0:
            print(f"Epoch: {epoch:03d} | Train loss: {np.mean(train_loss):.4f} | Val loss: {val_loss:.4f} | Train Accu: { np.mean(train_accu):.4f} | Val Accu: {np.mean(val_accu):.4f} | Train AUC: { np.mean(train_auc):.4f} | Val AUC: { np.mean(val_auc):.4f} ")


        #save best model
        if val_loss < best_loss:
            torch.save(model, DATA_PATH+f'/models/{model.__class__.__name__}_{model_name}.pt')
            best_loss = val_loss

    return performances_df