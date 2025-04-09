
import os
import random
import shutil
import unittest
from unittest.mock import patch, MagicMock
from split_train_test import SplitTrainTest

class TestSplitTrainTest(unittest.TestCase):
    """
    Function: Unittest that tests the SplitTrainTest class to validate the proper splitting of the dataset with as expected train-test ratio defined in the class.
              This is done by mocking the filesystem and checking the train-test ratio for individual classes and as well as on the whole data.
              In this test case, we are mocking the filesystem with 3 classes and 3 images in each class. 

    Note:
        Due to integer rounding, the exact 80-20 split might not be perfectly achieved, 
        especially with few images so the test uses assertAlmostEqual with a small delta of 0.2 for ratio comparisons.
    """
    
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.join')
    @patch('os.makedirs')
    @patch('shutil.copy')
    def test_split(self, mock_copy, mock_makedirs, mock_join, mock_isdir, mock_listdir,):
        # Mocking the directory structure
        data_path = "/mock/data"
        train_path = "/mock/train_data"
        test_path = "/mock/test_data"
        
        # Setup mock return values
        mock_listdir.side_effect = [
            ['class_1', 'class_2', 'class_3'], 
            ['img1.jpg', 'img2.jpg', 'img3.jpg'],  # class_1 images
            ['img1.jpg', 'img2.jpg', 'img3.jpg'],  # class_2 images
            ['img1.jpg', 'img2.jpg', 'img3.jpg']   # class_3 images
        ]
        
        mock_isdir.side_effect = lambda path: os.path.basename(path) in ['class_1', 'class_2', 'class_3']
        
        # Mocking os.path.join to return sensible paths
        def join_side_effect(*args):
            return "/".join(args)
        mock_join.side_effect = join_side_effect

        # Setting up an instance of SplitTrainTest with an 80-20 ratio
        split = SplitTrainTest(data_path=data_path, train_path=train_path, test_path=test_path, train_ratio=0.8)
        
        # Splitting the data
        split.split()
        
        # Verifying that the split is train-test ratio for each class during splitting.
        for idx, class_name in enumerate(['class_1', 'class_2', 'class_3']):
            total_images = 3
            expected_train_count = int(total_images * split.train_ratio)  # Should be 2 (3 * 0.8 -> int(2.4) = 2) if train ratio is 0.8
            expected_test_count = total_images - expected_train_count  # Should be 1
            
            # Checking whether the number of images copied for training and testing matches the expected count
            self.assertEqual(split.n_train_images[idx], expected_train_count, 
                            f"Train split for {class_name} is incorrect.")
            self.assertEqual(split.n_test_images[idx], expected_test_count, 
                            f"Test split for {class_name} is incorrect.")
        
        # Computing the number of train, test and total number of images.
        total_train,total_test = sum(split.n_train_images),sum(split.n_test_images)   # Should be 6 in train (2 per class), # Should be 3 in test(1 per class)
        total_images = total_train + total_test  # Should be 9
        
        overall_train_ratio = total_train / total_images  # Should be 6/9
        overall_test_ratio = total_test / total_images    # Should be 3/9
        
        # Note: The overall ratio won't be exactly 80-20 because of integer rounding
        # We should adjust our expectation or the test data
        self.assertAlmostEqual(overall_train_ratio, delta=0.2, 
                             msg="Overall train ratio is incorrect.")
        self.assertAlmostEqual(overall_test_ratio, 0.2, delta=0.2, 
                             msg="Overall test ratio is incorrect.")
if __name__ == '__main__':
    unittest.main()