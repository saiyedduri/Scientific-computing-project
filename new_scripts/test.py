# test.py
# Standard library imports
import os
import numpy as np

# Pytorch libraries
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader

# Importing functions and classes from split_train_test.py, plotting.py, preproessing.py
from preprocessing import load_dataloader
from model import SimplifiedResnet50
from plotting import plot_confusion_matrix, visualize_batch, min_max_image

def load_test_images(test_data_path, batch_size=32,shuffle=False):
    """
    Function: Load test images directly from the test data folder using ImageFolder
    
    Parameters:
        test_data_path (str): Path to the test data folder
        batch_size (int): Batch size for the dataloader
        shuffle(bool,optional): Parameter for shuffling the dataset 
        
    Returns:
        torch.utils.data.DataLoader: Dataloader for the test images
    """
    # Basic transforms - only resize and convert to tensor
    # Normalization will be applied later
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load the test dataset using ImageFolder
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=basic_transform)
    
    # Create a dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0)
    
    return test_loader, test_dataset.classes

def evaluate_model(checkpoint_path, test_data_path, shuffle_test=True,batch_size=32, visualize=True,save_dir=None,):
    """
    Function: Evaluate a trained model on test images and visualize results
    
    Parameters:
        checkpoint_path (str): Path to the model checkpoint
        test_data_path (str): Path to the test data folder
        save_dir (str): Directory to save results
        batch_size (int): Batch size for testing
        visualize (bool): Whether to visualize batch results
        
    Returns:
        tuple: (test_loss, test_accuracy, test_predictions, test_labels, batch_images, class_indices)
    """
    # device function For enabling support of cuda/nvidia device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # Loading the model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimplifiedResnet50(num_classes=10)  # Assuming 10 classes, will be adjusted if needed
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print("Model loaded successfully")
    
    # Loading the images directly from test data folder
    test_loader, class_names = load_test_images(test_data_path, batch_size,shuffle=shuffle_test)
    
    # Checking if classes match with expected number
    if hasattr(model, 'fc') and model.fc.out_features != len(class_names):
        print(f"WARNING: Model was trained for {model.fc.out_features} classes but found {len(class_names)} classes in test data")
    
    # Loading the training stats for normalization
    train_loader, train_global_stats = load_dataloader("train_dataloader.pth")
    
    # Creating a normalization transform using training stats
    normalize = transforms.Normalize(
        mean=train_global_stats['mean'].tolist(),
        std=train_global_stats['std'].tolist())
    
    # Storing the batch images for visualization
    all_batch_images = []
    all_class_indices = []
    all_predictions = []
    
    # Evaluating on the test set
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # Storing the original images for visualization before normalization
                if batch_idx == 0 and visualize:
                    # Convering to numpy for visualization later
                    batch_images_np = images.permute(0, 2, 3, 1).cpu().numpy()
                    all_batch_images = batch_images_np
                    all_class_indices = labels.cpu().numpy()
                
                # Applying the normalization
                normalized_images = torch.stack([normalize(img) for img in images])
                
                # Enabling GPU
                normalized_images, labels = normalized_images.to(device), labels.to(device)
                
                # Forward pass: Predicting class for each through optimization with loss function
                outputs = model(normalized_images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Predicting the label with maximum confidence and calculating the accuracy
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Store predictions for the visualization batch
                if batch_idx == 0 and visualize:
                    all_predictions = predicted.cpu().numpy()
    
        # Calculating the test accuracy metrics
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = (test_correct / test_total) * 100
        
        print(f"\nTest Results:")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Creating the confusion matrix if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cm_path = os.path.join(save_dir, "test_confusion_matrix.png")
            plot_confusion_matrix(
                labels=all_labels,
                predictions=all_preds,
                class_names=class_names,
                save_path=cm_path)
            print(f"Confusion matrix saved to {cm_path}")
        
        # Visualizing the batch of images with their predictions
        if visualize and len(all_batch_images) > 0:
            print("\nSample predictions:")
            for i in range(min(5, len(all_predictions))):
                pred_class = class_names[all_predictions[i]]
                true_class = class_names[all_class_indices[i]]
                print(f"Image {i+1}: Predicted: {pred_class}, Actual: {true_class}")
            
            if save_dir:
                # Save visualization using visualize_batch
                import matplotlib.pyplot as plt
                visualize_batch(all_batch_images, all_class_indices, class_names, min_max_image,predictions=all_predictions)
                fig=plt.savefig(os.path.join(save_dir, "sample_predictions.png"), bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"Sample predictions visualization saved to {save_dir}/sample_predictions.png")
    
    except Exception as e:
        print(f"Error during evaluation: {type(e).__name__} - {str(e)}")
        raise
        
    return avg_test_loss, test_accuracy, all_preds, all_labels, all_batch_images, all_class_indices, all_predictions, class_names

if __name__ == "__main__":
    # Set paths
    current_dir = os.getcwd()
    checkpoint_path = os.path.join(current_dir, "..", "results", "best_train_model.pth")
    test_data_path = os.path.join(current_dir, "..", "test_data")
    save_dir = os.path.join(current_dir, "..", "results", "plots")
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluating the model on the test data
    test_loss, test_acc, test_preds, test_labels, batch_images, class_indices, predictions, class_names = evaluate_model(
        checkpoint_path=checkpoint_path,
        test_data_path=test_data_path,
        shuffle_test=False,
        save_dir=save_dir,
        batch_size=32,
        visualize=True)
    
    print(f"Evaluation complete. Overall accuracy: {test_acc:.2f}%")