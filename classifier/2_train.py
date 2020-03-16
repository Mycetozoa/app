from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_dir = "/net/qnasimg/FURNITURE/okhrimenko/color_classifier/files#CL_COL_v6"
#data_dir = '/home/okhrimenko/Work/furniture/classifier/data/'
save_path = '/home/okhrimenko/Work/furniture/classifier/snapshots/'
gpu_id = 'cuda:1'

## Training options ##
batch_size = 16
num_epochs = 50
STEP_SIZE = 25
GAMMA = 0.1
input_size = 224
MOMENTUM = 0.9
LR = 0.005 


train_state = 'Training'    # need to change for resume training
snap_path = '/home/okhrimenko/Work/furniture/classifier/snapshots/snapshot_39.pth.tar'


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename="log.txt", level=logging.INFO, format=FORMAT)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    count = 0
    iteration_list = []
    #loss_list = []   #delete
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in np.arange(num_epochs) + start_epoch + 1:
        print('Epoch {}/{}'.format(epoch, num_epochs + start_epoch))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_path', 'test_path']:
            if phase == 'train_path':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train_path'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train_path':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train_path':
                        loss.backward()
                        optimizer.step()
                        count += 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # store loss and iteration         
                #loss_list.append(loss.data)
                #iteration_list.append(count)
            
                if count % 100 == 0:
                    #print(loss.item() * inputs.size(0))
                    print('Iteration {}, Loss {}'.format(count, loss.data))
                    logging.info('Iteration {}, Loss {}'.format(count, loss.data))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('{} Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test_path' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test_path':
                val_acc_history.append(epoch_acc)


        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_name = os.path.join(save_path, 'snapshot_{}.pth.tar'.format(epoch))
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'num_classes': num_classes,
            'class_to_idx': image_datasets['train_path'].class_to_idx,  
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        }, save_name)
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}, Epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


############## DATA ##############

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train_path': transforms.Compose([
        #transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size, input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_path': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train_path', 'test_path']}
num_classes = len(image_datasets['train_path'].classes)
print('num_classes:', num_classes)

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train_path', 'test_path']}


############## RUN ##############

# Detect if we have a GPU available
if torch.cuda.is_available():
    device = torch.device(gpu_id)
else:
    device = torch.device("cpu")

if train_state == 'Resume Training':
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    checkpoint = torch.load(snap_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    #scheduler.load_state_dict(checkpoint['scheduler'])
    scheduler = StepLR(optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
    optimizer_ft.load_state_dict(checkpoint['optimizer'])
    #start_epoch = checkpoint['epoch']
    start_epoch = 39
    #loss = checkpoint['loss']
else:
    start_epoch = 0
    # Initialize the model
    model = models.resnet50(pretrained=True)
    #model = models.googlenet(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Send the model to GPU
    model = model.to(device)

    # optimizer
    optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = StepLR(optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()



############ Train and evaluate
model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)
