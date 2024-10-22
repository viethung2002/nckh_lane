import torch
import torch.optim as optim
import numpy as np
import time
import copy
from model.lanenet.loss import compute_loss
from torch.utils.tensorboard import SummaryWriter
import os

def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type='FocalLoss', num_epochs=25, start_epoch=0, log_dir="logs"):
    since = time.time()
    training_log = {
        'epoch': [], 
        'training_loss': [], 
        'val_loss': [],
        'training_binary_loss': [], 
        'training_instance_loss': [],
        'training_binary_accuracy': [], 
        'training_binary_f1': [],
        'val_binary_loss': [], 
        'val_instance_loss': [],
        'val_binary_accuracy': [], 
        'val_binary_f1': []
    }
    
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Start training from start_epoch
    for epoch in range(start_epoch, num_epochs):
        training_log['epoch'].append(epoch)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0
            correct_binary = 0
            total_pixels = 0
            false_positive_result = 0
            true_positive_result = 0

            # Iterate over data
            for inputs, binarys, instances in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    # Kiểm tra cấu trúc của outputs
                    print(f"Epoch {epoch}, Phase: {phase}")
                    print(f"Model output type: {type(outputs)}")
                    
                    # Nếu outputs là list hoặc tuple
                    if isinstance(outputs, (list, tuple)):
                        print(f"Model output length: {len(outputs)}")
                        for i, output in enumerate(outputs):
                            print(f"Output {i} shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                    # Nếu outputs là dictionary
                    elif isinstance(outputs, dict):
                        print(f"Model output keys: {outputs.keys()}")
                        for key, output in outputs.items():
                            print(f"Output {key} shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                    # Nếu outputs là một tensor
                    elif isinstance(outputs, torch.Tensor):
                        print(f"Model output shape: {outputs.shape}")
                    else:
                        print(f"Unexpected model output type: {type(outputs)}")
                    
                    # Truy cập đầu ra dự đoán nhị phân từ outputs
                    binary_preds = torch.argmax(outputs['binary_seg_logits'], dim=1)
                    
                    # Sau khi kiểm tra cấu trúc, tiếp tục với tính toán loss
                    loss = compute_loss(outputs, binarys, instances, loss_type)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss[0].backward()
                        optimizer.step()

                # Statistics
                running_loss += loss[0].item() * inputs.size(0)
                running_loss_b += loss[1].item() * inputs.size(0)
                running_loss_i += loss[2].item() * inputs.size(0)

                # Binary accuracy and F1-Score calculation
                correct_binary += torch.sum(binary_preds == binarys).item()
                false_positive_result += torch.sum((binary_preds == 1) & (binarys == 0)).item()
                true_positive_result += torch.sum((binary_preds == 1) & (binarys == 1)).item()
                total_pixels += binarys.numel()

            # Calculate average losses and accuracies
            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            binary_accuracy = correct_binary / total_pixels

            # Precision, Recall, F1 calculations
            binary_total_false = total_pixels - correct_binary
            binary_precision = true_positive_result / (true_positive_result + false_positive_result) if (true_positive_result + false_positive_result) != 0 else 0
            binary_recall = true_positive_result / (true_positive_result + binary_total_false - false_positive_result) if (true_positive_result + binary_total_false - false_positive_result) != 0 else 0
            binary_f1_score = (2 * binary_precision * binary_recall) / (binary_precision + binary_recall) if (binary_precision + binary_recall) != 0 else 0

            print(f'{phase} Total Loss: {epoch_loss:.4f} Binary Loss: {binary_loss:.4f} Instance Loss: {instance_loss:.4f} Accuracy: {binary_accuracy:.4f} F1-Score: {binary_f1_score:.4f}')

            # Log metrics to TensorBoard
            writer.add_scalar(f"{phase} Binary Loss", binary_loss, epoch)
            writer.add_scalar(f"{phase} Instance Loss", instance_loss, epoch)
            writer.add_scalar(f"{phase} Binary Accuracy", binary_accuracy, epoch)
            writer.add_scalar(f"{phase} Binary F1-Score", binary_f1_score, epoch)

            # Save metrics to training log
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
                training_log['training_binary_loss'].append(binary_loss)
                training_log['training_instance_loss'].append(instance_loss)
                training_log['training_binary_accuracy'].append(binary_accuracy)
                training_log['training_binary_f1'].append(binary_f1_score)
            else:
                training_log['val_loss'].append(epoch_loss)
                training_log['val_binary_loss'].append(binary_loss)
                training_log['val_instance_loss'].append(instance_loss)
                training_log['val_binary_accuracy'].append(binary_accuracy)
                training_log['val_binary_f1'].append(binary_f1_score)

                # Deep copy the model if validation loss has decreased
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        torch.cuda.empty_cache()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val_loss: {best_loss:.4f}')

    # Save TensorBoard logs
    writer.flush()
    writer.close()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Convert lists to numpy arrays for easier handling later
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])
    training_log['training_binary_loss'] = np.array(training_log['training_binary_loss'])
    training_log['val_binary_loss'] = np.array(training_log['val_binary_loss'])
    training_log['training_instance_loss'] = np.array(training_log['training_instance_loss'])
    training_log['val_instance_loss'] = np.array(training_log['val_instance_loss'])
    training_log['training_binary_accuracy'] = np.array(training_log['training_binary_accuracy'])
    training_log['val_binary_accuracy'] = np.array(training_log['val_binary_accuracy'])
    training_log['training_binary_f1'] = np.array(training_log['training_binary_f1'])
    training_log['val_binary_f1'] = np.array(training_log['val_binary_f1'])

    return model, training_log
