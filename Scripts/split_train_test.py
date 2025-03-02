import os
import random
import shutil
from preprocessing import ImageTransformer,load_dataloader

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
                   For example, copies the images from the data class path to train/test paths

        Parameters:
            images (list): List of image file names to be copied.
            class_path (str): Path to the source directory (class folder).
            split_class_path (str): Path to the destination directory (train/test subfolder).
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
                train_class_path = os.path.join(self.train_path, class_name)
                test_class_path = os.path.join(self.test_path, class_name)
                # Creating class directory in each test and train directories
                os.makedirs(train_class_path, exist_ok=True)
                os.makedirs(test_class_path, exist_ok=True)

                # Shuffling the images present in each class directory of test and train folders
                class_images = os.listdir(class_path)
                random.shuffle(class_images)

                # Calculating the split index based on the train ratio.
                split_idx = int(len(class_images) * self.train_ratio)

                # Splitting the images into train and test based on the calculated split index
                train_images = class_images[:split_idx]
                test_images = class_images[split_idx:]
                self.n_train_images.append(len(train_images))
                self.n_test_images.append(len(test_images))

                # Storing the train images from data class path to the train class path in train folder.
                self.copy_dir(train_images, class_path, train_class_path)
                
                # Storing the test images from data class path to the train class path in the test folder.
                self.copy_dir(test_images, class_path, test_class_path)
                
        print(f"""Data is split into train and test sets successfully!
                The train data is stored at {self.train_path}
                The test data is stored at {self.test_path}""")

def main():
    # Defining data, train and test paths
    data_path = os.path.join(os.getcwd(), "..", "data_copy")
    train_path = os.path.join(os.getcwd(), "..", "train_data")
    test_path = os.path.join(os.getcwd(), "..", "test_data")

    # Creating an instance of SplitTrainTest and split the data
    split = SplitTrainTest(data_path=data_path, train_path=train_path, test_path=test_path, train_ratio=0.8)
    split.split()

    # Loading the batch_data_loader and retrieving the stored variables.
    loaded_dataloader, class_dict_stats, global_dict_stats=load_dataloader("batch_data_loader3.pth")
    dataset=loaded_dataloader.dataset

if __name__ == "__main__":
    main()