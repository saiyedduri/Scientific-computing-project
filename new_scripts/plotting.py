#plotting.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """
    Function: Visualzes and saves the confusion matrix in the save path, if provided
        The confusion matrix shows the count of predictions versus true labels, where:
        - Rows represent the true classes(actual labels).
        - Columns represent the predicted classes.
        - Diagonal cells indicate correct predictions.
        - Off-diagonal cells indicate misclassifications.
    Parameters:
            labels (list or np.array): True labels.
            predictions (list or np.array): Predicted labels of the model
            class_names (list[str]): List of class names corresponding to the labels.
            save_path (str, optional): Path to save the generated plot
    Returns:
        None: Displays the confusion matrix as a heatmap and optionally saves it to `save_path`.
    Example:
            >>> plot_confusion_matrix(best_labels,
                                      best_predictions,
                                      class_names=train_loader.dataset.classes,
                                      save_path="results/confusion_matrix.png")
    """
    cm = confusion_matrix(labels, predictions)
    
    # Creating a figure for the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted classes")
    plt.ylabel("True classes")
    
    # Saving the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, 
                save_loss_path=None, save_acc_path=None):
    """ 
    Function: Plots the training and validation losses and accuracy curves.
    Parameters:
        train_losses (list): Training loss values per epoch
        val_losses (list): Validation loss values per epoch (optional)
        train_accuracies (list): Training accuracy values (%) per epoch
        val_accuracies (list,optional): Validation accurloss and accuracy plots
    
    Example of usage:
    >>> plot_metrics(train_losses, val_losses, train_acc, val_acc)
    """
    #Plot loss
    plt.figure(figsize=(10, 5))

    plt.plot(train_losses, label="Training Loss", color='blue')
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_loss_path:
        plt.savefig(save_loss_path, bbox_inches='tight')

    # Plot accuracy and precision curves
    plt.figure(figsize=(10, 5))
    
    # Accuracy
    plt.plot(train_accuracies, label="Training Accuracy", color='green', linestyle='-')
    if val_accuracies is not None:
        plt.plot(val_accuracies, label="Validation Accuracy", color='green', linestyle='--')
    plt.title("Accuracy plot")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    if save_acc_path:
        plt.savefig(save_acc_path, bbox_inches='tight')

def min_max_image(image):
    """
    Function: Coverting the pixel values of the image of range [0,255] to the range of [0,1]

    The minimum and maximum of the image generated are not with in the range of 0 and 1. 
    So we are reducing the pixels values to the range of [0,1]

    Parameters:
        image(np.array): A numpy array of the image with pixel values ranging from [0,255]
    Returns:
        (np.array): A numpy array of the image with pixel values ranging from [0,1]
    """
    return (image - image.min()) / (image.max() - image.min())

def visualize_batch(images, labels, class_names, min_max_image_func,predictions=None):
    """
    Function:
        To visualize batch of images along with their corresponding labels in a grid.

    Parameters:
        images(np.array): Batch of images, of the shape (batch_size,height,width,channels)
        labels(np.array): class indices of the images
        class_names(List[str]):Class names of the images
        min_max_image_func:A function that takes an image (numpy array) as input and returns
                            the normalized image (in the range [0, 1]). This is typically used to
                            scale pixel values before visualization.
        predictions(np.array, optional): Predicted class indices of the images
     Returns:
        None: This function only handles visualization and does not return any values.
    
    Assumption: 
         Function assumes the images are in a batch and are represented as numpy arrays with shape 
          (batch_size, height, width, channels).
    Example use of the function:
        batch_images, class_indices = next(dataiter)
    
        # Converting the tensor to a numpy array and permute dimensions for matplotlib
        batch_images = batch_images.numpy()  # Convert to numpy array
        batch_images = batch_images.transpose(0, 2, 3, 1)  # Changing from (batch_size, channels, height, width) of Tensor to 
                                                            # (batch_size, height, width, channels) of numpy.

        class_names = transformed_data_class3.classes

        visualize_batch(batch_images, class_indices, class_names, min_max_image) 
    """
    # Calculating the grid size
    num_rows = int(np.ceil(np.sqrt(len(images))))
    
    # Creating subplots of the figure
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_rows, figsize=(10, 10))
    
    # Handling single-image case
    if num_rows == 1 and len(images) == 1:
        axes = np.array([axes])
    
    # Plotting the images in a grid
    for idx, axis in enumerate(axes.flatten()):
        if idx < len(images):
            # Reducing the range of the image to [0,1]
            normalized_image = min_max_image_func(images[idx])
            axis.imshow(normalized_image)  # Visualizing the image
            
            # Set the title with class name
            true_label = class_names[labels[idx]]
            title = f"{true_label}"
            
            # Add prediction if available
            if predictions is not None and idx < len(predictions):
                pred_label = class_names[predictions[idx]]
                if pred_label == true_label:
                    title += f"\nPred: {pred_label} ✓"
                    title_color = 'green'
                else:
                    title += f"\nPred: {pred_label} ✗"
                    title_color = 'red'
                axis.set_title(title, color=title_color)
            else:
                axis.set_title(title)
        
        axis.axis("off")
    
    plt.tight_layout()
    return fig 

def plot_number_of_images(stats_dict, save_title="barplot_number_of_images_per_class"):
    """
    Function: Plots and saves a bar chart representing the number of images per class using the class dictionary(class_dict)

    Parameters:
        stats_dict (dict): The dictionary returned by compute_class_stats, containing:
            - 'per_class': Dictionary with class-level statistics
            - 'global': Global statistics
        save_title (str): Title for the saved plot image.

    Returns:
        None: Saves the plot in the plots directory present in the  penultimate directory.
    
    Example usage of the function:
        plot_number_of_images(class_dict1,save_title="Original_dataset_number_of_images_per_class")

    """
    fig, axis = plt.subplots(figsize=(12, 8))
    
    # Extract per-class statistics
    class_stats = stats_dict["per_class"]
    
    # Preparing data for plotting
    classes = list(class_stats.keys())
    num_images = [class_stats[classname]["num_images"] for classname in classes]

    # Creating bar plot
    bars = axis.bar(classes, num_images)

    # Adding value labels
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width()/2., height + 3,
                 f'{num_images[idx]}',
                 ha='center', va='bottom', fontsize=10)
        
    # Setting axes and title of plot
    axis.set_xlabel("Class", fontsize=12)
    axis.set_ylabel("Number of Images", fontsize=12)
    axis.set_title("Number of images per class", fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # Creates the path in penultimate directory
    dir_path = os.path.join(os.getcwd(), "..", "plots")
    os.makedirs(dir_path, exist_ok=True)
    
    # Save and display
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, f"{save_title}.png"))
    print(f"Saved class distribution plot at {os.path.join(dir_path, f'{save_title}.png')}")
    plt.close()

