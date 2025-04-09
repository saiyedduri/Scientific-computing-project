#model.py
# Essential python modules
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Essential ML modules 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms

# Importing functions from preprocessing.py
from preprocessing import load_dataloader

class BottleneckBlock(nn.Module):
    """
    Function: A bottleneck block for ResNet-50 architecture.
    
    Parameters:
        class attribute:
            expansion (int): Expansion factor for the number of output channels
                            (fixed at 4 for ResNet-50).
        Instance attributes: 
            in_channels (int): Number of input channels
            out_channels (int):  Number of output channels before expansion
            stride (int, optional): Stride value for convolutional layers. By default: 1
            downsample (nn.Sequential, optional): Downsampling layer. By default: None
        Example of usage:
            block = BottleneckBlock(in_channels=64, out_channels=64, stride=1)
            x = torch.randn(2, 64, 56, 56)  # Batch of 2 samples
            output = block(x)  # Returns tensor of shape (2, 256, 56, 56)
    """
    expansion=4 # Defining expansion as class level attribute of bottle neck class

    def __init__(self, in_channels, out_channels, stride=1,downsample=None):
        super().__init__()
        # defining first convolution 1x1 kernel to reduce channels
        self.conv1=nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=1,
                             bias=False) # stride=1 is default
        self.batch_norm1=nn.BatchNorm2d(out_channels)
        
        # defining second convolution 3x3 kernel processes the feature maps.
        self.conv2=nn.Conv2d(out_channels,
                             out_channels,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             bias=False) 
        self.batch_norm2=nn.BatchNorm2d(out_channels)
        
        # defining third convolution(1x1) to expand channels
        self.conv3=nn.Conv2d(out_channels,
                             out_channels*self.expansion,
                             kernel_size=1,
                             bias=False)
        self.batch_norm3=nn.BatchNorm2d(out_channels* self.expansion)

        # defining downsample
        self.downsample = downsample
        # defining relu
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        """
        Function: Forward pass for the BottleneckBlock.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the block.
        """
        identity=x  # Input to each bootleneck block 

        out=self.conv1(x)
        out=self.batch_norm1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.batch_norm2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.batch_norm3(out)

        if self.downsample is not None:
            identity=self.downsample(x)
        
        out+=identity # Adding skip connection to the input, through identity mapping
        out=self.relu(out)

        return out

class SimplifiedResnet50(nn.Module):
    """
    Function:ResNet-50 model for image classification.

    Parameters:
        num_classes=number of classes considered for the model.
        in_channels (int): Number of input channels.
        conv1 (nn.Conv2d): Initial 7x7 convolution layer
        batch_norm1 (nn.BatchNorm2d): Batch normalization after the first convolution.
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool2d): Max pooling layer.
        res_layer1 (nn.Sequential): First residual layer.
        res_layer2 (nn.Sequential): Second residual layer.
        res_layer3 (nn.Sequential): Third residual layer.
        res_layer4 (nn.Sequential): Fourth residual layer.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Linear): Fully connected layer.
    """
    def __init__(self,num_classes=10):
        super().__init__()

        # Initializig the input channels
        self.in_channels = 64
        self.hyperparameters = {} 

        # Initial layer of 64 output channels with 7x7 kernel and stride 2. 
        self.conv1=nn.Conv2d(3,64,
                             kernel_size=7,
                             stride=2,
                             padding=3,
                             bias=False)
        # Batch normalizing the output channels of conv1
        self.batch_norm1=nn.BatchNorm2d(64)
        # Applying the ReLU, to introduce non-linearity  to convoluted output
        self.relu=nn.ReLU(inplace=True)
        #
        self.maxpool=nn.MaxPool2d(kernel_size=3,
                                  stride=2,
                                  padding=1)
        # Creating residual layers and initializing them
        self.res_layer1=self.residual_layer(64,3) # 3 blocks of Bottleneck with 64 output channels and stride 1
        self.res_layer2=self.residual_layer(128,4,stride=2) # 4 blocks of Bottleneck with 128 output channels and stride 2
        self.res_layer3=self.residual_layer(256,6,stride=2)# 6 blocks of Bottleneck with 256 output channels and stride 2
        self.res_layer4=self.residual_layer(512,3,stride=2)# 3 blocks of Bottleneck with 512 output channels and stride 2

        # Final layers: Average pooling 
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*BottleneckBlock.expansion,num_classes)
    
    def residual_layer(self,out_channels,blocks,stride=1):
        """
        Function:Creating a residual layer with multiple BottleneckBlocks.

        Parameters:
            out_channels (int): Number of output channels.
            blocks (int): Number of BottleneckBlocks in the layer.
            stride (int): Stride for the convolution.

        Returns:
            nn.Sequential: A sequential container of BottleneckBlocks.
        """
        # Renewing the downsample to None each time 
        downsample=None
        # Initializing to downsample if the output channels is not equal to input channels or stride>1
        if stride!=1 or self.in_channels!=out_channels*BottleneckBlock.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*BottleneckBlock.expansion,
                          kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*BottleneckBlock.expansion))

        # Renewing every residual layer and appending the set of bottlenecks
        residual_layers=[]
        residual_layers.append(BottleneckBlock(self.in_channels,
                                      out_channels,
                                      stride,
                                      downsample))
        
        # Updating input channels for the next residual layer by multiplying the outchannels by the exanpansion factor 4 for resnet-50
        self.in_channels=out_channels*BottleneckBlock.expansion
        
        # Appending the bootleneck blocks to form residual layer
        for _ in range(1,blocks):
            residual_layers.append(BottleneckBlock(
                self.in_channels,
                out_channels))
        
        return nn.Sequential(*residual_layers)
    
    def forward(self,x):
        """
        Function:Computing forward pass for the SimplifiedResnet50.

        Parameter:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        try:
            x=self.conv1(x)
            x=self.batch_norm1(x)
            x=self.relu(x)
            x=self.maxpool(x)

            x=self.res_layer1(x)

            x=self.res_layer2(x)

            x=self.res_layer3(x)

            x=self.res_layer4(x)

            x=self.avgpool(x)

            x=torch.flatten(x,1)

            x=self.fc(x)
            return x
        except RuntimeError as e:
            if "Input type" in str(e) or "shape" in str(e):
                print(f"Input Shape Error: Expected input format [Number of batches, Channels, Height, Wwight]")
                print(f"Received shape: {x.shape if x is not None else 'None'}")
            raise

def initialize_model(model,optimizer,learning_rate=0.01,device="cpu"):
    """
    Function: Initializes the model, loss function, and optimizer.

    Parameters:
        model (nn.Module): Model instance to initialize
        optimizer (str): Name of the optimizer ("SGD" or "Adam").
        learning_rate (float,optional): Learning rate for the optimizer. By dfault learning_rate=0.01
        device(str,optional): device ('cpu' or 'cuda') being used for training
        
    Returns: 
        tuple: (model, criterion, optimizer) initialized for training

    Example:
    >>> model = SimplifiedResnet50()
    >>> model, criterion, optimizer = initialize_model(
    >>>                                             model, optimizer="Adam", learning_rate=0.001, device="cuda")
    """
    try:
        model = model.to(device) 
        criterion = nn.CrossEntropyLoss()  

        if optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer== "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")
        
        return model, criterion, optimizer
    except Exception as e:
        print("Model initialization failed with error:", e)
        raise

def train_evaluate(model,train_batch_loader,
                    criterion,optimizer,num_epochs,device,val_batch_loader=None):
    """
    Function: To train and evaluates the model.

    Parameters:
        model (nn.Module): The neural network model.
        train_batch_loader (DataLoader): DataLoader for training data.
        val_batch_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim): Optimizer.
        num_epochs (int): Number of epochs to train.

    Returns:
        tuple: A tuple containing lists of training losses, training accuracies, validation losses, and validation accuracies.
        tuple: (train_losses (list), 
            train_accuracies (list), 
            val_losses (list), 
            val_accuracies (list))
    Example of usage:
        >>> train_losses, train_acc, val_loss, val_acc = train_evaluate(
                                                                    model, train_loader, criterion, optimizer,
                                                                    num_epochs=20, device="cuda", val_batch_loader=val_loader)
    """
    # Training Phase: Training the model
    train_losses=[]
    val_losses=[]
    train_accuracies=[]
    val_accuracies=[]
    val_preds = []
    val_labels = []
    num_classes=model.fc.out_features

    try:
        for epoch in range(num_epochs):

            model.train() # Setting the model to training mode.

            # Initializing the variables to track losses and accuracies during training.
            running_loss,train_correct,train_total=0.0,0,0
        
            for batch_images,labels in train_batch_loader:
                
                # Moving the batch data to the GPU(if available)
                batch_images,labels=batch_images.to(device),labels.to(device)
                
                optimizer.zero_grad() # Resetting the gradients to 0 for each batch.
                outputs=model(batch_images) # Forward pass: Computing the models predictions for the current batch
                loss=criterion(outputs,labels) # Computing the loss b/w models predictions and true labels
                loss.backward() # Backward pass: Computing the loss gradient w.r.t the model parameters
                optimizer.step() # Updates the model parameters

                running_loss+=loss.item() # Summing up the losses.
                _,predicted=torch.max(outputs.data,1)

                # Computing accuracy of the training data
                train_total +=labels.size(0) # Total number of samples in the batch.
                train_correct +=(predicted==labels).sum().item() # Counting the correct predictions.
                
            # Storing the training losses and accuracies after each epoch.
            train_losses.append(running_loss/len(train_batch_loader))
            train_accuracies.append((train_correct/train_total)*100)

            print(f"""Epoch {epoch + 1}/{num_epochs} - 
                    Train Loss: {train_losses[-1]:.4f}, 
                    Training Accuracy: {train_accuracies[-1]:.2f}%
                    """)

            # Validation Phase: Assesing the model performance on seperate subset of data that was not used in training data.
            #                   In other words, Validation is live tracking the model performance on unseen subset of the traning data.
            #                   We take different subset of the training data and validation data at every fold.
            if val_batch_loader is not None:
                model.eval() # Setting the model to evaluation mode()
                val_loss,val_correct,val_total=0.0,0,0
                all_preds = []
                all_labels = []
                with torch.no_grad(): # without updating the gradients
                    for batch_images,labels in val_batch_loader:
                        
                        batch_images,labels=batch_images.to(device),labels.to(device)
                        outputs=model(batch_images)  
                        loss=criterion(outputs,labels) # Computing the loss of the batch images
                        val_loss+=loss.item()
                        _,predicted=torch.max(outputs.data,1)
                        val_total+= labels.size(0)
                        val_correct+=(predicted==labels).sum().item()
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                            
                    # Storing the validation losses and accuracies after each epoch.
                    val_losses.append(val_loss/len(val_batch_loader))
                    val_accuracies.append((val_correct/val_total)*100)
                # Saving predictions and labels for the last epoch
                if epoch == num_epochs - 1:
                    val_preds = all_preds
                    val_labels = all_labels
                print(f"""Epoch {epoch + 1}/{num_epochs} - 
                        Validation Loss: {val_losses[-1]:.4f}, 
                        Validation Accuracy: {val_accuracies[-1]:.2f}%
                        """)

    except Exception as e:
        print(f"Training Error in train_evaluate function: {type(e).__name__} - {str(e)}")
        raise
    return train_losses, train_accuracies, val_losses, val_accuracies,val_preds, val_labels

def main():
    """
    # -----------------------------------------
    # Run the training pipeline in main execution
    # -----------------------------------------
    Function: End-to-end execution pipeline for model training.
    
    Workflow:
    1. Automatic device detection (GPU/CPU)
    2. Data loading from 'train_dataloader.pth'
    3. Model initialization with SGD optimizer
    4. Training for 10 epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Loading the train dataloader and validation dataloader.
    train_loader, _ = load_dataloader("train_dataloader.pth")
    
    # Initializing the model components like optimizer, learning rate and device
    model, criterion, optimizer = initialize_model(model=SimplifiedResnet50(num_classes=10),
                                                    optimizer="SGD",
                                                    learning_rate=0.01,
                                                    device=device)
    print("Model architecture:")
    print(model)

    # Train and evaluate the model
    train_losses, train_accuracies, val_losses, val_accuracies= train_evaluate(model= model,
                                                                                train_batch_loader=train_loader,
                                                                                criterion=criterion,
                                                                                optimizer=optimizer,
                                                                                num_epochs=10,
                                                                                device=device,
                                                                                val_batch_loader=None)
if __name__ == "__main__":
    main()

