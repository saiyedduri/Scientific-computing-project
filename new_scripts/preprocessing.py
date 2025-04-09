#preprocessing.py
# Essential python modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle
import shutil
import gc
gc.collect()
from PIL import Image

# Essential ML modules 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class ImageTransformer(Dataset):
    """
    The class loads the images from a images directory and applies optional transformations(including data augmentation),
    and allows for storing the augmented images back into respective class directories.
    It is assumed that the filenames in the "images_path" directory are unique to avoid the conflict.
    
    Methods:
    __len__(): Returns the number of image paths in the dataset.
    __getitem__(idx): Returns an image and its corresponding class index after applying the necessary transformations for idx(index) in the batch passed to the function.
    save_transformed_image(image_tensor, original_image_path, class_idx): Saves an augmented image back to the respective class folder.

    Example for acessing ImageTransformer:
        transforms1=transf`orms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(global_stats1["mean"].tolist(), global_stats1["std"].tolist())
                                        ])
    
        transformed_data_class1 = ImageTransformer(
                                                    images_path=os.path.join(curr_dir,"..","data_copy"),
                                                    fixed_size=(224,224)
                                                    transform=transforms1
                                                    augment_data=False
                                                    )
    """
    def __init__(self, images_path, fixed_size,transform=transforms.Compose([transforms.ToTensor()]),
                 augment_data=False,save_dir=None,random_seed=42):
        """
        Function: Initializes the ImageTransformer class with the given parameters.

        Parameters:
            images_path (str): The root directory containing image subdirectories for each class.
            fixed_size (tuple): The target size to which all images will be resized.
            transform (callable, optional): A function/transform to apply to the image. Default transformation of transforms.ToTensor() is applied on
                                            each image of batch.
            augment_data (bool, optional): If True, applies data augmentation to the images. By default it's set to False.
            save_dir(str,optional)      :Path at which newly transformed images are stored. By default None,indicating no save directory will be created for saving images.
            random_seed (int, optional): Random seed for reproducibility. By deafult the random_seed is set to 42.
            classes (list): List of class names found in images_path
            image_paths (list): List of (image_path, class_index) tuples

        """

        self.images_path = images_path
        self.fixed_size=fixed_size
        self.transform = transform
        self.augment_data=augment_data
        self.classes = os.listdir(images_path)  
        self.image_paths = []
        self.random_seed=random_seed
        self.save_dir=save_dir
        
        # Creating the directory if the directory doesnt exist
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Setting the random seed to make random operations more reproducible while data augmentation, cpu operations
        random.seed(self.random_seed) #  random module makes RandomHorizontalFlip,RandomRotation more reproducible
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed) # Sets the random seed for Pytorch CPU operations
        torch.cuda.manual_seed(self.random_seed) # Sets the random seed for Pytorch single GPU operations.
        
        # Creating a list of image paths and corresponding labels
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(images_path, class_name)
            if os.path.isdir(class_path):
                print(f"Scanning through {class_name} images.......")
                for filename in os.listdir(class_path):                              
                    image_path = os.path.join(class_path, filename) # Add the file name to the class_path
                    #Checking the image path is a file(not directory) and 
                    #Adding the image_path to the image_paths list
                    if os.path.isfile(image_path):
                        self.image_paths.append((image_path, class_idx))

    def __len__(self):
        """
        Function:Returns the total number of image paths in the dataset.

        Returns:
            (int): The number of image paths in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Function: Returns an image and its corresponding class index after applying the necessary transformations for idx(index) in the batch passed to the function.

        Parameters:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: 
                image (Tensor): The transformed image as a tensor.
                                   If data augmentation is enabled, the augmented image is returned.
                class_idx (int): The index of the image's class.
        """
        image_path, class_idx = self.image_paths[idx]

        # Loading each image and converting to 'RGB' format
        image = Image.open(image_path).convert('RGB')

        # resizing the every image to get into a fixed size
        image = image.resize(self.fixed_size)
        
        if self.augment_data:
            # Applying the transformations
            augmented_image=self.transform(image)

            if self.save_dir:
                # Saves the transformed PIL image to each class in the save_dir path as transformed_{imagename}
                transformed_image_path=self.save_image(augmented_image, image_path, class_idx, prefix="transformed")

            return augmented_image,class_idx

        else:
            # Converts just to tensor and return for non-data augmented images
            image = self.transform(image)

            if self.save_dir:
                # Save the original resized image
                _ = self.save_image(image, image_path, class_idx, prefix="original")
            
            return image, class_idx
    
    def save_image(self, image_tensor, original_image_path,class_idx,prefix):
        """
        Function: Saves the transformed images through data augmentation to the respective class folder.
                  The function returns the saved file path to __get__item() while processing each image.
        
        Parameters:
            image_tensor (Tensor): The transformed image tensor to save.
            original_image_path (str): The path of the original image before transformation.
            class_idx (int): The index of the class to save the transformed image in.
        Returns:
            (str): The file path where the transformed image has been saved.   
        """
        class_name=self.classes[class_idx]

        # Creating the class directory inside the save directory
        class_save_dir=os.path.join(self.save_dir,class_name)
        if not os.path.exists(class_save_dir):
            os.makedirs(class_save_dir)

        # Saving the image to the class folder with original_{imagename} if the image is original, else saved as transformed_{imagename}
        save_path=os.path.join(class_save_dir,f"{prefix}_{os.path.basename(original_image_path)}")
        
        # Converting the tensor back to PIL image
        save_image(image_tensor,save_path)

        return save_path

def compute_class_stats(dataloader, dataset):
    """
    Function: Computes per-class and global statistics (mean, standard deviation, max, min) of the images of the dataset by
              computing mean, standard deviation, maximum and minimum across red, blue and green channels of each image.

              This is done through  loading batches of images by dataloader.
    Parameters:
        dataloader (torch.utils.data.DataLoader): The data loader that provides the batches of images and corresponding labels.
        dataset (torch.utils.data.Dataset): The dataset containing the image classes and other metadata (like class names).
    Returns:
        dict: A tuple containing:
           - 'per_class': Class-wise statistics (mean, std, sum, etc.)
            - 'global': Dataset-wide statistics (mean, std) of red, blue and green 
    Example:
        class_dict_stats, global_dict_stats = compute_class_stats(dataloader, dataset)
        
        Example of accessing class-level stats: 
        print(class_dict_stats['cat']['mean_redpixels'])  # Mean value of red channel for 'cat' class
        print(class_dict_stats['dog']['std_bluepixels'])  # Standard deviation of blue channel for 'dog' class
            
    """
    class_stats = {}
    global_stats = {
        "sum_red": 0.0,
        "sum_green": 0.0,
        "sum_blue": 0.0,
        "sum_sq_red": 0.0,
        "sum_sq_green": 0.0,
        "sum_sq_blue": 0.0,
        "total_pixels": 0}

    # Accumulating sums for both class and global stats
    for batch_images, labels in dataloader:
        for img, label in zip(batch_images, labels):
            # Converting the image to shape [C, H*W]
            img_flat = img.view(3, -1)
            red_channel = img_flat[0]
            green_channel = img_flat[1]
            blue_channel = img_flat[2]
            
            # Get class name from dataset
            class_idx = label.item()
            class_name = dataset.classes[class_idx]
            
            # Initializing the class stats
            if class_name not in class_stats:
                class_stats[class_name] = {
                    "sum_red": 0.0,
                    "sum_green": 0.0,
                    "sum_blue": 0.0,
                    "sum_sq_red": 0.0,
                    "sum_sq_green": 0.0,
                    "sum_sq_blue": 0.0,
                    "num_images": 0,
                    "total_pixels": 0
                }
            
            # Updating the class statistics
            class_stats[class_name]["sum_red"] += red_channel.sum()
            class_stats[class_name]["sum_green"] += green_channel.sum()
            class_stats[class_name]["sum_blue"] += blue_channel.sum()
            
            class_stats[class_name]["sum_sq_red"] += (red_channel ** 2).sum()
            class_stats[class_name]["sum_sq_green"] += (green_channel ** 2).sum()
            class_stats[class_name]["sum_sq_blue"] += (blue_channel ** 2).sum()
            
            class_stats[class_name]["num_images"] += 1
            class_pixels = red_channel.numel()
            class_stats[class_name]["total_pixels"] += class_pixels
            
            # Updating the global statistics
            global_stats["sum_red"] += red_channel.sum()
            global_stats["sum_green"] += green_channel.sum()
            global_stats["sum_blue"] += blue_channel.sum()
            
            global_stats["sum_sq_red"] += (red_channel ** 2).sum()
            global_stats["sum_sq_green"] += (green_channel ** 2).sum()
            global_stats["sum_sq_blue"] += (blue_channel ** 2).sum()
            
            global_stats["total_pixels"] += class_pixels

    # Computing the per-class means and standard deviations
    for class_name in class_stats:
        cs = class_stats[class_name]
        total_pixels = cs["total_pixels"]
        
        # Means
        cs["mean_red"] = cs["sum_red"] / total_pixels
        cs["mean_green"] = cs["sum_green"] / total_pixels
        cs["mean_blue"] = cs["sum_blue"] / total_pixels
        
        # Standard deviations of the images
        cs["std_red"] = torch.sqrt((cs["sum_sq_red"] / total_pixels) - (cs["mean_red"] ** 2))
        cs["std_green"] = torch.sqrt((cs["sum_sq_green"] / total_pixels) - (cs["mean_green"] ** 2))
        cs["std_blue"] = torch.sqrt((cs["sum_sq_blue"] / total_pixels) - (cs["mean_blue"] ** 2))

    # Computing the means and standard deviations of the images
    global_mean = torch.tensor([
        global_stats["sum_red"] / global_stats["total_pixels"],
        global_stats["sum_green"] / global_stats["total_pixels"],
        global_stats["sum_blue"] / global_stats["total_pixels"]
    ])
    
    global_std = torch.tensor([
        torch.sqrt((global_stats["sum_sq_red"] / global_stats["total_pixels"]) - (global_mean[0] ** 2)),
        torch.sqrt((global_stats["sum_sq_green"] / global_stats["total_pixels"]) - (global_mean[1] ** 2)),
        torch.sqrt((global_stats["sum_sq_blue"] / global_stats["total_pixels"]) - (global_mean[2] ** 2))
    ])

    return {
        "per_class": class_stats,
        "global": {
            "mean": global_mean,
            "std": global_std
        }
    }

def save_dataloader(dataloader,stats_dict,filename):
    """
    Function: Saves the DataLoader and its associated dataset with the current random state for reproducibility.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The DataLoader to be saved.
        stats_dict (dict): Full statistics from compute_class_stats
        filename (str): The path to the file where the DataLoader will be saved.
    Example of usage:
        dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4)
        stats_dict = compute_class_stats(dataloader)
        save_dataloader(dataloader, stats_dict, "saved_dataloader.pth")
    """
    # Saving random states for reproducibility
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    dataset = dataloader.dataset
    # Saving random states for reproducibility
    save_dict={
        "dataset":dataset,
        "stats_dict": stats_dict,
        "random_state":random_state,
        "np_state":np_state,
        "torch_state":torch_state,
        "batch_size":dataloader.batch_size,
        "num_workers":dataloader.num_workers}
    try:
        with open(filename,"wb") as file:
            pickle.dump(save_dict,file)
    except Exception as e:
        print("Error loading data:", e)
        raise
    print(f"Dataloader and  random state saved as {filename}")

def load_dataloader(filename):
    """
    Function: Loads the saved DataLoader and restores the random states 

    Parameters:
        filename (str): The path to the file where the DataLoader was saved.

    Returns:
        DataLoader: The reloaded DataLoader.
    Example of usage:
        dataloader, stats_dict = load_dataloader("saved_dataloader.pth")
    
    """
    # loading the saved dictionary
    try:
        with open(filename, 'rb') as f:
            saved_data = pickle.load(f)
    except Exception as e:
        print("Error loading data:", e)
        raise

    # loading dataset and random states
    loaded_dataset = saved_data['dataset']
    global_dict_stats = saved_data['stats_dict']["global"]
    
    # loading random states for reproducibility
    random.setstate(saved_data['random_state'])
    np.random.set_state(saved_data['np_state'])
    torch.random.set_rng_state(saved_data['torch_state'])

    # Recreating the DataLoader
    loaded_dataloader = DataLoader(
        loaded_dataset,
        batch_size=saved_data['batch_size'],
        num_workers=saved_data['num_workers'])

    print(f"DataLoader loaded from {filename}")
    return loaded_dataloader, global_dict_stats



