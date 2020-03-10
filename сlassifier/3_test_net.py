from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
from PIL import Image
from torch.autograd import Variable

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#test_dir = '/home/sarmnv/ROSINKA/old_classifier_pytorch/test_data/schweppes'
#test_dir = '/home/okhrimenko/Work/furniture/classifier/data'
test_dir = '/net/qnasimg/FURNITURE/okhrimenko/furniture_classifier/data#CL_SHAPE_v7/test_path/'
#weights_path = '/home/okhrimenko/Work/furniture/classifier/snapshots_MAT_resnet50/snapshot_34.pth.tar'
weights_path = '/home/okhrimenko/Work/furniture/classifier/files#CL_SHAPE_v8/snapshot_17.pth.tar'

batch_size = 32
input_size = 224
gpu_id = 'cuda:1'
inference_to_folder = True
test_out = '/home/okhrimenko/Work/furniture/classifier/test_out'


def imshow(inp, path, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    #plt.imshow(inp)
    plt.imsave(path, inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

start_time = time.time()
############## DATA ##############
data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create test dataset
image_dataset = datasets.ImageFolder(test_dir, data_transform)

# Create test dataloader
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


device = torch.device(gpu_id)
############## RUN ##############
checkpoint = torch.load(weights_path)

class_to_idx = checkpoint['class_to_idx']
idx_to_class = {val: key for key, val in class_to_idx.items()}
num_classes = len(class_to_idx)
print(len(class_to_idx))

# Initialize the model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()


fig = plt.figure()
running_corrects = 0
with torch.no_grad():
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        if inference_to_folder is True:
            for j in range(inputs.size()[0]):
                label = idx_to_class[int(preds[j])]
                out_path = os.path.join(test_out, label, '{}_{}.jpg'.format(idx, j))
                if os.path.exists(os.path.join(test_out, label)) is False:
                    os.mkdir(os.path.join(test_out, label))
                imshow(inputs.cpu().data[j], out_path)
 
        running_corrects += torch.sum(preds == labels.data)
        
    
accuracy = running_corrects.double() / len(dataloader.dataset)
print('Acc: {:.4f}'.format(accuracy))
#print(confusion_matrix)
#print(confusion_matrix.diag()/confusion_matrix.sum(1))
print("Execution time: --- %s seconds ---" % (time.time() - start_time))
