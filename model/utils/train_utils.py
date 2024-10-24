import torch
import torch.nn.functional as F
from tqdm import tqdm
from model.lanenet.loss import compute_loss  # Import the custom loss function
import numpy as np
from model.eval_function import calculate_map
from model.scnn.loss import SCNNLoss
from model.laneatt.loss import lossLaneatt

# General training function for both LaneNet and SCNN models
def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type, num_epochs, start_epoch, compute_loss_fn):
    """
    General training function that can be used for both LaneNet and SCNN models.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        scheduler: Learning rate scheduler.
        dataloaders (dict): Dictionary containing 'train' and 'val' dataloaders.
        dataset_sizes (dict): Dictionary containing the sizes of the training and validation datasets.
        device (torch.device): Device to run the training on (CPU or GPU).
        loss_type: The type of loss function used.
        num_epochs (int): Total number of epochs to train.
        start_epoch (int): Epoch to start training from.
        compute_loss_fn (function): Loss computation function for the model.

    Returns:
        model: The best model after training.
        log: Dictionary containing training logs.
    """
    best_model_wts = None
    best_val_loss = float('inf')
    log = {'epoch': [], 'training_loss': [], 'val_loss': [], 'val_map': []}

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_map = 0.0

            # Iterate over data.
            with tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch}') as pbar:
                for inputs, binarys, instances in dataloaders[phase]:
                    inputs = inputs.to(device)
                    binarys = binarys.to(device)
                    instances = instances.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        total_loss, binary_loss, instance_loss, binary_seg_pred = compute_loss_fn(outputs, binarys, instances, loss_type)

                        # Backward pass + optimize only if in training phase
                        if phase == 'train':
                            total_loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += total_loss.item() * inputs.size(0)

                    # Calculate mAP for validation phase
                    if phase == 'val':
                        binary_preds = torch.sigmoid(binary_seg_pred)  # Apply sigmoid to get probability
                        mean_ap = calculate_map(binary_preds, binarys)
                        running_map += mean_ap * inputs.size(0)

                    pbar.update(1)

            # Compute average loss for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            log[f'{phase}_loss'].append(epoch_loss)

            # Calculate average mAP for validation phase
            if phase == 'val':
                epoch_map = running_map / dataset_sizes[phase]
                log['val_map'].append(epoch_map)
                print(f'{phase} Loss: {epoch_loss:.4f} mAP: {epoch_map:.4f}')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f}')

            # Save the best model weights for validation phase
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = model.state_dict()

        log['epoch'].append(epoch)

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, log

# Specific training functions for LaneNet and SCNN  
def train_lanenet_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type, num_epochs, start_epoch):
    """
    Wrapper function for training the LaneNet model.
    """
    return train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type, num_epochs, start_epoch, compute_loss)

def train_scnn_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type, num_epochs, start_epoch):
    """
    Wrapper function for training the SCNN model.
    Note: If SCNN has a different loss function, you can define a specific one and pass it here.
    """
    return train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type, num_epochs, start_epoch, SCNNLoss)
def train_laneatt_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, start_epoch=0):
    """
    Wrapper function for training the LaneATT model.
    """
    return train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type=None, num_epochs=num_epochs, start_epoch=start_epoch, compute_loss_fn=lossLaneatt)
