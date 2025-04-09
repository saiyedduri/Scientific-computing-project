# main.py
# Standard library imports
import os
import warnings
import argparse

# Pytorch libraries
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# Importing functions and classes from split_train_test.py, plotting.py, preproessing.py,hyperparametertuning.py
from split_train_test import SplitTrainTest
from hyperparametertuning import TransformedSubset, HyperparameterTuning
from preprocessing import load_dataloader, compute_class_stats, ImageTransformer
from model import SimplifiedResnet50, initialize_model, train_evaluate
from plotting import plot_confusion_matrix, plot_metrics

class ModelTrainingPipeline:
    """
    Function: A pipeline to handle model training, hyperparameter tuning, retraining, and evaluation.
    """
    def __init__(self, data_root=None):
        """
        Function: Initializes the ModelTrainingPipeline.

        Instance attributes:
            device (torch.device): Specifies whether to use GPU or CPU.
            data_root (str): Root directory for the dataset.
            model (torch.nn.Module): The deep learning model.
            criterion (torch.nn.modules.loss._Loss): Loss function.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
        
        Example of usage:
            >>> pipeline = ModelTrainingPipeline(data_root="/path/to/data")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root if data_root else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._setup_paths()
        self._create_directories()
        self.model = None
        self.criterion = None
        self.optimizer = None

    def _setup_paths(self):
        """Initialize all file paths"""
        self.data_path = os.path.join(self.data_root, "data_copy")
        self.train_path = os.path.join(self.data_root, "train_data")
        self.test_path = os.path.join(self.data_root, "test_data")
        self.results_dir = os.path.join(self.data_root, "results")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.best_hyperparameter_model_path = os.path.join(self.results_dir, "best_hyperparametertuning_model.pth")
        self.final_best_model_path = os.path.join(self.results_dir, "best_train_model.pth")

    def _create_directories(self):
        """Function:Creating required output directories"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

    def prepare_data(self,aug_transform,train_ratio=0.8):
        """
        Function: Splits the dataset with test data and augmented training data.
                  Creates dataloaders for training and testing.
        
        Example:
            >>> pipeline = ModelTrainingPipeline()
                # Data augmentation the train data using data augmentation techniques
            >>> my_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.ToTensor()])
            >>> pipeline.prepare_data(aug_transform=my_transform)
        """
        print("\nSplitting dataset and creating dataloaders...")
        split = SplitTrainTest(
            data_path=self.data_path,
            train_path=self.train_path,
            test_path=self.test_path,
            train_ratio=train_ratio)
        split.split()
        
        # Applying augmentation to the training data and saving training augmented images to the existing training directory.
        print("\nGenerating augmented images...")
        augmented_dataset = ImageTransformer(
            images_path=self.train_path,
            fixed_size=(224, 224),
            transform=aug_transform,
            augment_data=True,  
            save_dir=self.train_path)

        aug_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=False)
        # Saving the augmented images to the training path along with the original images per class
        for _ in aug_loader:
            pass

        split.create_and_save_dataloaders(batch_size=32)
    
    def tune_hyperparameters(self, param_grid):
        """
        Function: Performs hyperparameter tuning to find the best model parameters.
        Parameters:
            param_grid (dict): Dictionary specifying hyperparameter search space.

        Example of usage:
            >>> pipeline = ModelTrainingPipeline()
            >>> param_grid = {"learning_rates": [0.001, 0.01], "batch_sizes": [32, 64]}
            >>> pipeline.tune_hyperparameters(param_grid)
        """
        print("\nStarting hyperparameter tuning...")
        try:
            train_loader, _ = load_dataloader("train_dataloader.pth")
            self.tuner = HyperparameterTuning(
                model=SimplifiedResnet50,
                num_classes=len(train_loader.dataset.classes)  ,
                train_dataset=train_loader.dataset,
                device=self.device)
            
            best_params, best_performances = self.tuner.tune(param_grid)
            
            self.tuner.plot_training_results(self.plots_dir)
        except Exception as e:
            print(f"Fatal error in hyperparameter tuning main: {str(e)}")
            raise
    def initialize_best_model(self,checkpoint_path):
        """ 
        Function: Loads and initializes the best model based on tuning results.
        Parameters:
            checkpoint_path: Pickle file Hyperparamertuning model with 
        Example of usage.
            >>> pipeline = ModelTrainingPipeline()
            >>> pipeline.initialize_best_model()
        """
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        self.model = SimplifiedResnet50(num_classes= checkpoint["best_hyperparameters"]["num_classes"])
        
        self.model.hyperparameters = { "batch_size": checkpoint["best_hyperparameters"]["batch_size"],
                                    "num_epochs": checkpoint["best_hyperparameters"]["num_epochs"],
                                    "learning_rate": checkpoint["best_hyperparameters"]["learning_rate"]}
    
        self.model, self.criterion, self.optimizer = initialize_model(model=self.model,
                                                                        optimizer=checkpoint["best_hyperparameters"]["optimizer"],
                                                                        learning_rate=checkpoint["best_hyperparameters"]["learning_rate"],
                                                                        device=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    def retrain_full_model(self, save_best_model=True):
        """
        Function: Retrains the model on the full training dataset using the best hyperparameters.

        Parameters:
            save_best_model (bool, optional): Whether to save the retrained model. Defaults to True.
        Example:
            >>> pipeline = ModelTrainingPipeline()
            >>> pipeline.retrain_full_model()
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_best_model() first.")

        # Loading the training data and stats
        train_loader, train_global_stats = load_dataloader("train_dataloader.pth")
        
        # Normalizing the full training data
        normalized_train = TransformedSubset( train_loader.dataset, 
                                              transform=transforms.Normalize(mean=train_global_stats['mean'].tolist(),
                                                                                std=train_global_stats['std'].tolist()))
        
        # Creating the DataLoader with best batch size (already loaded from checkpoint)
        full_train_loader = DataLoader(normalized_train,
                                        batch_size=self.model.hyperparameters["batch_size"],
                                        shuffle=True)
        
        # training the model on full training data
        train_losses, train_accuracies, _, _, _, _ = train_evaluate(
            model=self.model,
            train_batch_loader=full_train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            num_epochs=self.model.hyperparameters["num_epochs"],
            device=self.device,
            val_batch_loader=None)
        if save_best_model:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "hyperparameters": 
                    {
                    "batch_size": self.model.hyperparameters["batch_size"],
                    "num_epochs": self.model.hyperparameters["num_epochs"],
                    "learning_rate": self.model.hyperparameters["learning_rate"],
                    "optimizer": type(self.optimizer).__name__
                    },
                    "train_losses": train_losses,
                    "train_accuracies": train_accuracies
                    }, self.final_best_model_path)
        
        plot_metrics(train_losses=train_losses,
                    val_losses=None,
                    train_accuracies=train_accuracies,
                    val_accuracies=None,
                    save_loss_path=os.path.join(self.plots_dir, "full_train_loss_curve.png"),
                    save_acc_path=os.path.join(self.plots_dir, "full_train_accuracy_curve.png"))
                                
# -------------------------------------------------------
# Run the training and testing pipeline in main execution
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Hyperparameter Tuning')
    parser.add_argument('--lr', nargs='+', type=float, required=True,
                       help='Learning rates (space-separated)')
    parser.add_argument('--bs', nargs='+', type=int, required=True,
                       help='Batch sizes (space-separated)')
    parser.add_argument('--epochs', nargs='+', type=int, required=True,
                       help='Training epochs (space-separated)')
    parser.add_argument('--optim', type=str, required=True, choices=['SGD', 'Adam'],
                       help='Optimizer type')
    parser.add_argument('--k_folds', type=int, required=True,
                       help='Number of CV folds')
    parser.add_argument('--retrain', type=str, default=None,
                   help='Path to best hyperparameter checkpoint for retraining')
    parser.add_argument('--skip_tuning', action='store_true',
                   help='Skip hyperparameter tuning if retraining')
    
    args = parser.parse_args()

    pipeline = ModelTrainingPipeline()
    
    # Data augmentation the train data using data augmentation techniques
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()])
    
    if not args.skip_tuning:
        # Splitting and preparaing the data into 80% train and 20% test data
        pipeline.prepare_data(aug_transform,train_ratio=0.8)

        # Hyperparametertuning phase
        if not all([args.lr, args.bs, args.epochs, args.optim, args.k_folds]):
            raise ValueError("All tuning parameters (--lr, --bs, --epochs, --optim, --k_folds) required when not skipping tuning")
    
        # Validating the  input parameters
        if not all(lr > 0 for lr in args.lr):
            raise ValueError("Learning rates must be positive")
        if not all(bs > 0 for bs in args.bs):
            raise ValueError("Batch sizes must be positive")

        # Building the  parameter grid
        param_grid = {"learning_rates": args.lr,
            "batch_sizes": args.bs,
            "num_epochs": args.epochs,
            "optimizers": [args.optim],
            "k_folds": args.k_folds}

        # Executing the hyperparameter tuning
        pipeline.tune_hyperparameters(param_grid)
        
        # Generating the config ID for saving
        config_id = (f"LR_{'_'.join(map(str, args.lr))}_"
                    f"BS_{'_'.join(map(str, args.bs))}_"
                    f"EP_{'_'.join(map(str, args.epochs))}_"
                    f"{args.optim}")
        
        # Saving best model from tuning
        pipeline.tuner.save_best_model(os.path.join(pipeline.results_dir, f"best_model_{config_id}.pth"))
    
    # Retraining phase
    if args.retrain:
        if not os.path.exists(args.retrain):
            raise FileNotFoundError(f"Checkpoint file {args.retrain} not found. Example: python new_main.py --retrain ./results/best_model_LR_0.001_BS_32_EP_20_Adam.pth --skip_tuning")
    
        # Initializing the best model
        pipeline.initialize_best_model(args.retrain)
        
        # Retraining with loaded parameters
        pipeline.retrain_full_model(save_best_model=True)


