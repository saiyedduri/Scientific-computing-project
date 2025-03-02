import os
import random
import shutil
import unittest
from unittest.mock import patch
from preprocessing import ImageTransformer, load_dataloader
from split_train_test import SplitTrainTest

class TestSplitTrainTest(unittest.TestCase):
    def setUp(self):
        # Creat a temporary directory structure for testing
        self.data_path = "test_data"
        self.train_path = "test_train"
        self.test_path = "test_test"
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

        # Create a mock dataset with 2 classes and 10 images each
        self.class_names = ["class1", "class2"]
        self.num_images_per_class = 10
        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            for i in range(self.num_images_per_class):
                with open(os.path.join(class_path, f"image_{i}.jpg"), "w") as f:
                    f.write("mock image data")

    def tearDown(self):
        # Clean up the temporary directories after the test
        shutil.rmtree(self.data_path)
        shutil.rmtree(self.train_path)
        shutil.rmtree(self.test_path)

    def test_split_train_test(self):
        # Initialize the SplitTrainTest object
        split = SplitTrainTest(self.data_path, self.train_path, self.test_path, train_ratio=0.8)
        split.split()

        # Verify that the train and test directories are created correctly
        for class_name in self.class_names:
            train_class_path = os.path.join(self.train_path, class_name)
            test_class_path = os.path.join(self.test_path, class_name)
            self.assertTrue(os.path.exists(train_class_path))
            self.assertTrue(os.path.exists(test_class_path))

            # Verify the number of images in train and test directories
            train_images = os.listdir(train_class_path)
            test_images = os.listdir(test_class_path)
            self.assertEqual(len(train_images), 8)  # 80% of 10 images
            self.assertEqual(len(test_images), 2)  # 20% of 10 images

        # Verify the n_train_images and n_test_images lists
        self.assertEqual(split.n_train_images, [8, 8])  # 8 train images per class
        self.assertEqual(split.n_test_images, [2, 2])  # 2 test images per class

    @patch("preprocessing.load_dataloader")
    def test_train_test_ratios(self, mock_load_dataloader):
        # Mock the dataset and dataloader
        mock_dataset = list(range(20))  # Mock dataset with 20 samples
        mock_dataloader = type('MockDataLoader', (), {'dataset': mock_dataset})
        mock_load_dataloader.return_value = (mock_dataloader, {}, {})

        # Initialize the SplitTrainTest object
        split = SplitTrainTest(self.data_path, self.train_path, self.test_path, train_ratio=0.8)
        split.split()

        # Calculate and verify the train and test ratios
        train_ratio = sum(split.n_train_images) / len(mock_dataset)
        test_ratio = sum(split.n_test_images) / len(mock_dataset)
        self.assertAlmostEqual(train_ratio, 0.8, places=2)  # 80% train ratio
        self.assertAlmostEqual(test_ratio, 0.2, places=2)  # 20% test ratio

if __name__ == "__main__":
    unittest.main()