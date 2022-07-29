from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True

torch.manual_seed(17)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],) 


image_size = 100

data_transforms = {
    'Five-Training': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Modified-Testing': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = '/Users/bradyjohnsen/Desktop/LMIC Internship/BodyPartProject/Datasets/'
data_dir = '~/bodypartproject/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['Modified-Testing']}

#training_data_dir = '/Users/bradyjohnsen/Desktop/LMIC Internship/BodyPartProject/Datasets/Five-Training'
training_data_dir = '~/bodypartproject/Five-Training'

image_datasets['Five-Training'] = ImageFolderWithPaths(
    root=training_data_dir, transform=data_transforms['Five-Training'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['Modified-Testing']}

dataloaders['Five-Training'] = torch.utils.data.DataLoader(image_datasets['Five-Training'], batch_size=100,
                                              shuffle=True, num_workers=0)

dataset_sizes = {x: len(image_datasets[x])
                 for x in ['Five-Training', 'Modified-Testing']}
class_names = image_datasets['Five-Training'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Five-Training', 'Modified-Testing']:
            if phase == 'Five-Training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'Five-Training':
                for inputs, labels, paths in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Five-Training'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'Five-Training':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'Five-Training':
                    scheduler.step()
            else:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Five-Training'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'Five-Training':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'Five-Training':
                    scheduler.step()

            if phase == 'Five-Training':
                epoch_loss = running_loss / (dataset_sizes[phase] - 1)
                epoch_acc = running_corrects.double() / (dataset_sizes[phase] - 1)

            else:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'Modified-Testing' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)

    print()

    print(f'Best val Acc: {best_acc:4f}')

    add_grid = [global_pathlist[idx_exclu], best_acc.item()]

    print(add_grid)

    grid.append(add_grid)

    torch.save(model_ft.state_dict(), f"{imagelist[idx_exclu]}-{best_acc:4f}.pth")

    print()
    print()

    return model


rows, cols = (dataset_sizes['Five-Training'], 2)
grid = [[0]*cols]*rows

print(f"Dataset size: {dataset_sizes['Five-Training']}")

for inputs, labels, paths in dataloaders['Five-Training']:
   tlabels = labels
   tpaths = paths 

global_labels = labels
global_pathlist = tpaths

imagelist = []

for idx in range(len(global_pathlist)):
    imagelist.append(global_pathlist[idx].rsplit('/', 1)[-1])

print(imagelist)

for idx_exclu in range(dataset_sizes['Five-Training']):

    image_datasets['Five-Training'] = ImageFolderWithPaths(
    root=training_data_dir, transform=data_transforms['Five-Training'])

    image_datasets['Five-Training'].imgs.remove((global_pathlist[idx_exclu], global_labels[idx_exclu]))

    print()
    print(f'Excluding: {global_pathlist[idx_exclu]} of class {class_names[global_labels[idx_exclu]]}')

    dataloaders['Five-Training'] = torch.utils.data.DataLoader(image_datasets['Five-Training'], batch_size=100,
                                              shuffle=True, num_workers=0)

    print()

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=5)

print(grid)
