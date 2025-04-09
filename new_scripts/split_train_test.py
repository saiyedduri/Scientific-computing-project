#split_train_test.py
# Essential python modules
import os
import random
import shutil

# Essential ML modules 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Importing functions from preprocessing.py
from preprocessing import ImageTransformer, compute_class_stats, save_dataloader

class SplitTrainTest:
    def __init__(self, data_path, train_path, test_path, train_ratio=0.8):
        """
        Function:Initializes the SplitTrainTest object with the given paths for dataset, training, and testing directories, 
        and a train ratio for splitting the data.
        
        Parameters:
            data_path (str): The path to the source dataset.
            train_path (str): The path to the output training directory.
            test_path (str): The path to the output testing directory.
            train_ratio (float): Ratio of the data to be used for training (default 0.8).
        
        Example usage:
            split = SplitTrainTest(data_path="dataset", train_path="train_data", test_path="test_data", train_ratio=0.8)
        """
        self.data_path = data_path
        self.train_path = train_path
        self.test_path = test_path
        self.train_ratio = train_ratio
        self.n_train_images = []
        self.n_test_images = []

    def copy_dir(self, images, class_path, split_class_path):
        """
        Function: Copies a list of image files from the source directory to the destination directory.
                   For example, copies the images from the data class path to train/test paths.
        Parameters:
            images (list): List of image file names to be copied.
            class_path (str): Path to the source directory (class folder).
            split_class_path (str): Path to the destination directory (train/test subfolder).
        Example usage:
            split.copy_dir(["image1.jpg", "image2.jpg"], "dataset/classA", "train_data/classA")
        """
        for file_name in images:
            source_path = os.path.join(class_path, file_name)
            destination_path = os.path.join(split_class_path, file_name)
            shutil.copy(source_path, destination_path)

    def split(self):
        """
        Function: Splits the dataset into training and testing sets based on the specified train ratio.
        It shuffles the images within each class, divides them into train and test sets, and
        copies them into their respective directories.
        """
        for class_name in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                train_class_path,test_class_path  = os.path.join(self.train_path, class_name),os.path.join(self.test_path, class_name)

                # Creating class directory in each test and train directories
                os.makedirs(train_class_path, exist_ok=True)
                os.makedirs(test_class_path, exist_ok=True)

                # Shuffling the images present in each class directory of test and train folders
                class_images = os.listdir(class_path)
                random.shuffle(class_images)

                # Calculating the split index based on the train ratio.
                split_idx = int(len(class_images) * self.train_ratio)

                # Splitting the images into train and test based on the calculated split index.
                train_images,test_images = class_images[:split_idx],class_images[split_idx:]

                self.n_train_images.append(len(train_images))
                self.n_test_images.append(len(test_images))

                # Storing the train images from data class path to the train class path in train folder.
                self.copy_dir(train_images, class_path, train_class_path)
                
                # Storing the test images from data class path to the train class path in the test folder.
                self.copy_dir(test_images, class_path, test_class_path)
                
        print(f"""Data is split into train and test sets successfully!
                The train data is stored at {self.train_path}
                The test data is stored at {self.test_path}""")
        
    def create_and_save_dataloaders(self,batch_size=32):
        """
        Function:
        Creates and saves the train and test dataloaders for un-normalized data. 
        Initially,images are resized and transformed to tensor and then converted to train and test loaders.

        Parameters:
        batch_size(int): Number of images to be considered per batch for train and test dataloaders.
        
        Example usage:
            split.create_and_save_dataloaders(batch_size=64)
        """
        # Initializing the train and test datasets
        train_dataset = ImageTransformer(
            images_path=self.train_path,
            fixed_size= (224, 224),
            transform=transforms.Compose([transforms.ToTensor()]),
            augment_data=False)
        
        test_dataset = ImageTransformer(
            images_path=self.test_path,
            fixed_size= (224, 224),
            transform=transforms.Compose(
                [transforms.ToTensor()]),
            augment_data=False)

        # Creating train dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0)
        
        # Creating test dataloaders
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0)
        print("Computing global and per-class statistics of augmented dataset ")
        # Compute statistics from training data only
        stats_dict= compute_class_stats(train_loader, train_dataset)
        
        print("Saving dataloaders with statistics")
        # Save dataloaders with computed statistics
        save_dataloader(
            train_loader,
            stats_dict,
            "train_dataloader.pth")
        
        save_dataloader(
            test_loader,
            stats_dict,  # Passsing the  train stats  for normalization of test data
            "test_dataloader.pth")
        print("Train and test dataloaders are saved.")

def main():
    # Defining data, train and test paths
    data_path = os.path.join(os.getcwd(), "..", "data_copy")
    train_path = os.path.join(os.getcwd(), "..", "train_data")
    test_path = os.path.join(os.getcwd(), "..", "test_data")

    # Creating an instance of SplitTrainTest and split the data
    split = SplitTrainTest(data_path=data_path, train_path=train_path, test_path=test_path, train_ratio=0.8)
    split.split()

    # Defining data augmentation transformations
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])
    
    # Applying augmentation to the training data and saving training augmented images to the existing training directory.
    print("\nGenerating augmented images...")
    augmented_dataset = ImageTransformer(
        images_path=train_path,
        fixed_size=(224, 224),
        transform=aug_transform,
        augment_data=True,  
        save_dir=train_path)

    aug_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=False)
    # Saving the augmented images to the training path 
    for _ in aug_loader:
        pass

    # Saving the augmented images dataloaders with the given batch size.
    split.create_and_save_dataloaders(batch_size=32)

if __name__ == "__main__":
    main()
