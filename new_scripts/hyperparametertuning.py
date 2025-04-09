#hyperparametertuning.py
# # Essential python modules
import os
import pandas as pd
import matplotlib.pyplot as plt

# Essential ML modules
from sklearn.model_selection import KFold
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, Dataset

# Importing classes and functions from model.py and preprocessing.py
from model import (
    SimplifiedResnet50,
    initialize_model,
    train_evaluate)
from plotting import plot_metrics,plot_confusion_matrix
from preprocessing import load_dataloader, compute_class_stats

class TransformedSubset(Dataset):
    """
    Function: Applies transformations to a subset of data.

    Parameters:
        subset (torch.utils.data.Dataset): Original dataset subset
        transform (callable, optional): Transform to apply to subset items
    
    Returns:
        for each batch, the class returns
            x: Transformed subset of the dataset
            y: class index of the dataset
        (int): Returns the length of the subset to access.

    Example of usage:
        subset = Subset(dataset, indices)
        transformed = TransformedSubset(subset, transform=transforms.RandomHorizontalFlip())
        loader = DataLoader(transformed, batch_size=32)
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    

class HyperparameterTuning:
    """
    Function:Performs hyperparameter tuning with k-fold cross validation.

    Parameters:
        model (nn.Module class): Model class to tune (uninitialized)
        num_classes (int): Number of output classes
        train_dataset (Dataset): Full training dataset
        device (str): Compute device ('cpu' or 'cuda')
    
    Example of usage:
        tuner = HyperparameterTuning(
            model=SimplifiedResnet50,
            num_classes=10,
            train_dataset=train_data,
            device="cpu")
        best_params, metrics = tuner.tune(param_grid)
    """
    def __init__(self, model,num_classes,train_dataset, device="cpu"):
        self.model=model
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.class_names = train_dataset.classes
        self.best_parameters = None
        self.best_performances = None
        self.best_model_state=None
        self.all_foldresults = pd.DataFrame()
        self.device = device


    def _kfold_cross_validation(self,params):
        """
        Function: To Perform k-fold cross-validation on the model with the given params
        Parameters:
            params (dict): Hyperparameters to validate containing:
                - learning_rate (float): Learning rate for the optimizer.
                - batch_size (int):Batch size for DataLoader.
                - num_epochs (int): Number of epochs to train.
                - optimizer (str):Name of the optimizer ("SGD" or "Adam")
                - k_folds (int):  Number of folds for cross-validation.
        Returns:
            pd.DataFrame: A DataFrame containing the results of each fold with
                          training, validation losses and accuracies at each epoch,
        """
        kfold=KFold(n_splits=params["k_folds"],shuffle=True,random_state=42)
        fold_results = []
        all_val_preds = []
        all_val_labels = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.train_dataset)):
            try:
                print(f"Fold {fold+1}/{params['k_folds']}")

                # Creating subsets of training and validation data
                train_subset=Subset(self.train_dataset,train_idx)
                val_subset=Subset(self.train_dataset,val_idx)

                # Computing the mean and std from the training subset per fold
                temp_train_loader = DataLoader(train_subset, 
                                                batch_size=params["batch_size"],
                                                shuffle=False)
                stats = compute_class_stats(temp_train_loader, self.train_dataset)

                # Defining normalization transform
                normalize = transforms.Normalize(mean=stats['global']['mean'].tolist(), 
                                                std=stats['global']['std'].tolist())
                # Applying normalization to both subsets
                normalized_train = TransformedSubset(train_subset, transform=normalize)
                normalized_val = TransformedSubset(val_subset, transform=normalize)

                # Creating the dataloaders with normalized data
                train_loader = DataLoader(normalized_train, batch_size=params["batch_size"], shuffle=True)
                val_loader = DataLoader(normalized_val, batch_size=params["batch_size"], shuffle=False)

                # Initializing the model and train for each fold
                model = self.model(num_classes=self.num_classes)
                model,criterion,optimizer=initialize_model(model=model,
                                                            optimizer=params["optimizer"],
                                                            learning_rate=params["learning_rate"],
                                                            device=self.device)
                train_losses, train_accuracies, val_losses, val_accuracies, val_preds, val_labels=train_evaluate(model=model,
                                                                                                                    train_batch_loader=train_loader,
                                                                                                                    criterion=criterion,
                                                                                                                    optimizer=optimizer,
                                                                                                                    num_epochs=params["num_epochs"],
                                                                                                                    device=self.device,
                                                                                                                 val_batch_loader=val_loader)
                # Aggregating predictions and labels
                all_val_preds.extend(val_preds)
                all_val_labels.extend(val_labels)
                            
                # Appending the results of each epoch to the foldresults dataframe 
                for epoch in range(params['num_epochs']):
                    fold_results.append({
                            **params,
                            "fold": fold + 1,
                            "epoch": epoch + 1,
                            "train_loss": train_losses[epoch],
                            "train_accuracy": train_accuracies[epoch],
                            "val_loss": val_losses[epoch],
                            "val_accuracy": val_accuracies[epoch]})
            except Exception as e:
                print(f"Error in Fold {fold+1}: {str(e)}")
                continue # continue tuning after the error
        return pd.DataFrame(fold_results), model.state_dict().copy(), all_val_preds, all_val_labels

    def tune(self,param_grid):
            """
            Function:Executes grid search over the parameter combinations.

            Parameters:
                param_grid (dict): Hyperparameter search space with keys:
                    - learning_rates (list[float]): list of learning rates for the optimizer.
                    - batch_sizes (list[int]): list of batch sizes for DataLoader.
                    - num_epochs (list[int]): list of number of epochs to train.
                    - optimizers (list[str]): list of optimizers 
                    - k_folds (int): Number of folds of training data
            Returns:
                tuple: (best_parameters (dict): Optimal parameter set,
                        best_performances (dict): Validation metrics for best model)
            
            Example of usage:
                tuner = HyperparameterTuning(model=SimplifiedResnet50,
                                            num_classes=10,
                                            train_dataset=train_data,
                                            device="cpu")
                param_grid = {
                    "learning_rates": [0.001, 0.01],
                    "batch_sizes": [32, 64],
                    "num_epochs": [10, 20],
                    "optimizers": ["SGD", "Adam"],
                    "k_folds": 5
                }
                best_params, metrics = tuner.tune(param_grid)
            """ 
            best_accuracy=0
            param_combinations=list(itertools.product(param_grid['learning_rates'],
                                                    param_grid['batch_sizes'],
                                                    param_grid['num_epochs'],
                                                    param_grid['optimizers']))
            for learning_rate,batch_size,num_epochs,optimizer in param_combinations:
                    try:
                        print(f"Evaluating model with learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs},optimizer={optimizer}")
                        params = {'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'num_epochs': num_epochs,
                                    'optimizer': optimizer,
                                    'k_folds': param_grid.get('k_folds', 5),
				                    'num_classes': self.num_classes}
                        
                        results_df, model_state, val_preds, val_labels=self._kfold_cross_validation(params)
                        self.all_foldresults = pd.concat([self.all_foldresults, results_df])
                        
                        # Collecting the final epoch accuracy of every fold
                        final_epoch_accuracies=results_df.groupby("fold")["val_accuracy"].last()
                        avg_val_accuracy=final_epoch_accuracies.mean()

                        print(f"   Average validation accuracy:{avg_val_accuracy:.3f}%\n")

                        # Finding the best validation accuracy by comparision with previous best in the loop
                        if avg_val_accuracy>=best_accuracy:
                            best_accuracy=avg_val_accuracy
                            self.best_parameters = params
                            self.best_model_state = model_state
                            self.best_performances={
                                    "train_loss":results_df["train_loss"],
                                    "train_accuracy":results_df["train_accuracy"],
                                    "val_loss":results_df["val_loss"],
                                    "val_accuracy":results_df["val_accuracy"],
                                    "batch_size":batch_size,
                                    "num_epochs": num_epochs,
                                    "val_preds": val_preds,
                                    "val_labels": val_labels}
                    except Exception as e: 
                        print(f"Critical error in combination {params}:")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Message: {str(e)}\n")
                        continue # Skips the combination and proceeds to the next.
                
            print(f"Best hyperparameters found: {self.best_parameters}")
            print(f"Best validation accuracy: {best_accuracy:.2f}%")
            return self.best_parameters, self.best_performances
    def save_best_model(self, save_path):
        """
        Function: Saves the best model's state_dict and hyperparameters.
        Parameters:
        save_path(str): The path at which best model file to be saved.
        
        Example of usage:
            tuner = HyperparameterTuning(model=SimplifiedResnet50,
                            num_classes=10,
                            train_dataset=train_data,
                            device="cpu")
            tuner.save_best_model("best_model.pth")
        """
        if self.best_model_state is None:
            raise ValueError("No model to save. Run tune() first.")
        
        # Create directory if missing
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        safe_all_foldresults = self.all_foldresults.to_dict()
        # Saving model state
        torch.save({
            "model_state_dict": self.best_model_state,
            "best_hyperparameters": self.best_parameters,
            "best_performances": self.best_performances,
            "all_foldresults": self.all_foldresults

        }, save_path)
        print(f"Best model saved to {save_path}")    

    def plot_training_results(self,save_dir):
        """
        Function: 
            Plots the results for traning
             -Visualizes and saves the training/validation metrics for best parameter combination in the save_dir path,
             -Saves the best performances and best parameters in to the save_dir path.
           
        Parameters:
            save_dir: Path to save the accuracy plots, best performances and directory
        
        Example of usage:
                tuner = HyperparameterTuning(model=SimplifiedResnet50,
                num_classes=10,
                train_dataset=train_data,
                device="cpu")
                tuner.plot_training_results("results/plots")

        """
        if self.best_performances is None:
            print("No results to plot. Run tune() first.")
            return

        # Creating directory if missing
        os.makedirs(save_dir, exist_ok=True)
        
        # Calling plotting with save paths
        plot_metrics(self.best_performances['train_loss'],
            self.best_performances['val_loss'],
            self.best_performances['train_accuracy'],
            self.best_performances['val_accuracy'],
            save_loss_path=os.path.join(save_dir, "loss_curves.png"),
            save_acc_path=os.path.join(save_dir, "accuracy_curves.png"))
        
        plot_confusion_matrix(labels=self.best_performances["val_labels"],
                            predictions=self.best_performances["val_preds"],
                            class_names=self.class_names,
                            save_path=os.path.join(save_dir, "confusion_matrix.png"))
        
def main():
    """
    Function:
        # -----------------------------------------
        # Run the hyperparametertuning pipeline in main execution
        # -----------------------------------------
        Main execution pipeline for hyperparameter tuning workflow.
        1. Initializes compute device
        2. Loads training data
        3. Configures parameter grid
        4. Performs grid search with cross-validation
        5. Visualizes best model metrics
    """
    try:
        # Initializing the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Loading the dataLoader of the augmented dataset
        train_loader, _ = load_dataloader("train_dataloader.pth")

        # Performing hyperparameter tuning
        tuner = HyperparameterTuning(model=SimplifiedResnet50,
                                    num_classes=10,
                                    train_dataset=train_loader.dataset,
                                    device=device)

        param_grid = {
            "learning_rates": [0.001, 0.01],
            "batch_sizes": [32, 64],
            "num_epochs": [10, 20],
            "optimizers": ["SGD", "Adam"],
            "k_folds": 3}
        
        best_parameters, best_performances =tuner.tune(param_grid)

        
        # Saving the best model with best parameters and best performance metrics
        tuner.save_best_model(save_path=os.path.join(os.getcwd(), "..", "results", "best_model.pth"))

        # Plotting the accuracy metrics(training losses, validation losses)
        tuner.plot_training_results(save_dir=os.path.join(os.getcwd(), "..", "results","plots"))

    except Exception as e:
        print(f"Fatal error in hyperparameter tuning main execution: with error {str(e)}")
        raise

if __name__ == "__main__":
    main()



