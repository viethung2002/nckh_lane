import torch
import torch.optim as optim
import numpy as np
import time
import copy
from model.csnn.loss import CSNNLoss

def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type='CrossEntropyLoss', num_epochs=25, start_epoch=0):
    since = time.time()
    training_log = {'epoch': [], 'training_loss': [], 'val_loss': []}
    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())

    # Khởi tạo hàm loss dành riêng cho CSNN
    loss_fn = CSNNLoss(loss_type=loss_type)

    # Bắt đầu huấn luyện từ start_epoch
    for epoch in range(start_epoch, num_epochs):
        training_log['epoch'].append(epoch)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Mỗi epoch có train và validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model sang chế độ training
            else:
                model.eval()  # Set model sang chế độ evaluation

            running_loss = 0.0

            # Lặp qua dữ liệu
            for inputs, binarys, _ in dataloaders[phase]:  # Chỉ cần binarys cho CSNN
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)

                # Reset optimizer gradient
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    # Tính toán loss cho CSNN
                    loss = loss_fn(outputs, binarys)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Thống kê loss
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Cập nhật best model nếu là validation phase
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                training_log['training_loss'].append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val_loss: {best_loss:.4f}')

    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log
 