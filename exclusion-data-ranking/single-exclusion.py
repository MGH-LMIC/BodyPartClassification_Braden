from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

num_epochs = 5
image_size = 100

#data_dir = '/Users/bradyjohnsen/Desktop/LMIC Internship/BodyPartProject/Datasets/'
data_dir = '~/bodypartproject/'

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.manual_seed(17)
np.random.seed(0)

since_entire = time.time()

data_transforms = {
    'full-training': transforms.Compose([
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

#def seed_worker(worker_id):
    #worker_seed = torch.initial_seed() % 2**32
    #np.random.seed(worker_seed)
    #random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['full-training', 'Modified-Testing']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=0, generator=g)
               for x in ['Modified-Testing']}

dataloaders['full-training'] = torch.utils.data.DataLoader(image_datasets['full-training'], batch_size=100,
                                              shuffle=True, num_workers=0, generator=g)

dataset_sizes = {x: len(image_datasets[x])
                 for x in ['full-training', 'Modified-Testing']}
class_names = image_datasets['full-training'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_sum = 0.0

    acc_arr = np.zeros(num_epochs)
    loss_arr = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['full-training', 'Modified-Testing']:
            if phase == 'full-training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'full-training':
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'full-training'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'full-training':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'full-training':
                    scheduler.step()
            else:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'full-training'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'full-training':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'full-training':
                    scheduler.step()

            if phase == 'full-training':
                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

            else:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                acc_arr[epoch] = epoch_acc
                loss_arr[epoch] = epoch_loss
                acc_sum = acc_sum + epoch_acc

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

    avg_acc = (acc_sum/num_epochs)

    print()
    print(f'Best val Acc: {best_acc:4f}')
    print()
    print(f"Average val Acc: {avg_acc}")
    print()
    print(f"Accuracy: {acc_arr}")
    print()
    print(f"Testing Loss: {loss_arr}")
    print()

    return model, best_acc, avg_acc 

grid = []

print(f"Training dataset size: {dataset_sizes['full-training']}")
print(f"Testing dataset size: {dataset_sizes['Modified-Testing']}")
print()

global_labels = []
global_pathlist = []
imagelist = []

for i in range(dataset_sizes['full-training']):
    fname, label = image_datasets['full-training'].samples[i]
    global_pathlist.append(fname)
    global_labels.append(label)
    imagelist.append(global_pathlist[i].rsplit('/', 1)[-1])

#print(imagelist)
print()

for index in range(dataset_sizes['full-training']):

    print(f"Index: {index}")
    print()

    image_datasets['full-training'] = datasets.ImageFolder(os.path.join(data_dir, 'full-training'), transform=data_transforms['full-training'])

    add_grid = []

    image_datasets['full-training'].imgs.remove((global_pathlist[index], global_labels[index]))
    print(f"Removed \"{imagelist[index]}\"")
    add_grid.append(imagelist[index])

    print()

    dataloaders['full-training'] = torch.utils.data.DataLoader(image_datasets['full-training'], batch_size=50,
                                              shuffle=True, num_workers=0, generator=g)

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

    model_ft, best_accuracy, avg_accuracy = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=num_epochs)

    torch.save(model_ft.state_dict(), f"{index}-{best_accuracy:4f}.pth")

    add_grid.append(best_accuracy.item())
    add_grid.append(avg_accuracy.item())

    grid.append(add_grid)
    print(add_grid)

    print()
    print()

print(grid)

np.savetxt("grid-single.csv", 
           grid,
           delimiter =", ", 
           fmt ='% s')

print("CSV saved. Making control model.")

image_datasets['full-training'] = datasets.ImageFolder(os.path.join(data_dir, 'full-training'), transform=data_transforms['full-training'])
dataloaders['full-training'] = torch.utils.data.DataLoader(image_datasets['full-training'], batch_size=50,
                                              shuffle=True, num_workers=0, generator=g)
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

model_ft, best_accuracy, avg_accuracy = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=num_epochs)

torch.save(model_ft.state_dict(), f"Control-{best_accuracy:4f}.pth")

time_elapsed_complete = time.time() - since_entire
print(f"Ranking complete in {time_elapsed_complete // 60:.0f}m {time_elapsed_complete % 60:.0f}s")