import torch
import torch.nn.functional as F
from tqdm import tqdm
from model.lanenet.loss import compute_loss  # Import the custom loss function
import numpy as np

# Function to calculate mAP
def calculate_map(binary_preds, binary_labels, thresholds=np.linspace(0.0, 1.0, 11)):
    """Calculate mean Average Precision (mAP) based on multiple thresholds."""
    precision_at_thresholds = []
    
    for threshold in thresholds:
        # Apply threshold
        binary_preds_thresholded = (binary_preds >= threshold).float()

        # Calculate true positives, false positives, false negatives
        tp = torch.sum((binary_preds_thresholded == 1) & (binary_labels == 1)).item()
        fp = torch.sum((binary_preds_thresholded == 1) & (binary_labels == 0)).item()
        fn = torch.sum((binary_preds_thresholded == 0) & (binary_labels == 1)).item()

        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-6) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn + 1e-6) if (tp + fn) != 0 else 0

        # Store precision value at this threshold
        precision_at_thresholds.append(precision)
    
    # Calculate mean Average Precision
    mean_ap = np.mean(precision_at_thresholds)
    return mean_ap

# Define the training function for LaneNet
def train_lanenet_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type, num_epochs, start_epoch):
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
                        total_loss, binary_loss, instance_loss, binary_seg_pred = compute_loss(outputs, binarys, instances, loss_type)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            log[f'{phase}_loss'].append(epoch_loss)

            # Calculate average mAP for validation phase
            if phase == 'val':
                epoch_map = running_map / dataset_sizes[phase]
                log['val_map'].append(epoch_map)
                print(f'{phase} Loss: {epoch_loss:.4f} mAP: {epoch_map:.4f}')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f}')

            # Deep copy the model if it has the best loss in validation
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = model.state_dict()

        log['epoch'].append(epoch)

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, log

# Define the training function for CSNN
def train_csnn_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type, num_epochs, start_epoch):
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
                        total_loss, binary_loss, instance_loss, binary_seg_pred = compute_loss(outputs, binarys, instances, loss_type)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            log[f'{phase}_loss'].append(epoch_loss)

            # Calculate average mAP for validation phase
            if phase == 'val':
                epoch_map = running_map / dataset_sizes[phase]
                log['val_map'].append(epoch_map)
                print(f'{phase} Loss: {epoch_loss:.4f} mAP: {epoch_map:.4f}')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f}')

            # Deep copy the model if it has the best loss in validation
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = model.state_dict()

        log['epoch'].append(epoch)

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, log
