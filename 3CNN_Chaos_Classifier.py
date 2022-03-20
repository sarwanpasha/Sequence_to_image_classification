from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import time

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pandas as pd
from numpy import mean


##############################
#       flatten 2d list to 1d
#############################
def flatten(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


####################################################
#       Create Train, Valid and Test sets
####################################################
train_data_path = './chaos_oreder_minimizer_train_test_20k/train'
test_data_path = './chaos_oreder_minimizer_train_test_20k/test'

train_image_paths = []
classes = []

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[1])
print('class example: ', classes[1])

train_image_paths, valid_image_paths = train_image_paths[:int(0.8 * len(train_image_paths))], train_image_paths[
                                                                                              int(0.8 * len(
                                                                                                  train_image_paths)):]

train_classes = []
validate_classes = []
for i in range(len(train_image_paths)):
    train_classes.append(train_image_paths[i].split('/')[-2])
for i in range(len(valid_image_paths)):
    validate_classes.append(valid_image_paths[i].split('/')[-2])

test_image_paths = []
test_classes = []
for data_path in glob.glob(test_data_path + '/*'):
    test_classes.append(data_path.split('/')[-1])
    test_image_paths.append(glob.glob(data_path + '/*'))
test_image_paths = list(flatten(test_image_paths))

test_classes = []
for i in range(len(test_image_paths)):
    test_classes.append(test_image_paths[i].split('/')[-2])

print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths),
                                                             len(test_image_paths)))
train_values, train_counts = np.unique(train_classes, return_counts=True)
validate_values, validate_counts = np.unique(validate_classes, return_counts=True)
test_values, test_counts = np.unique(test_classes, return_counts=True)


#######################################################
#      Create dictionary for class indexes
#######################################################
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}


#######################################################
#               Define Dataset Class
#######################################################
class CovidImagesDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transforms.Compose([transforms.Resize(480),
                                       transforms.ToTensor(),
                                       ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        img = Image.open(image_filepath).convert('RGB')
        img = np.array(img)
       # img = cv2.resize(img, (256, 256))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        return img, label


#######################################################
#                  Create Dataset
#######################################################
train_dataset = CovidImagesDataset(train_image_paths)
valid_dataset = CovidImagesDataset(valid_image_paths)  # test transforms are applied
test_dataset = CovidImagesDataset(test_image_paths)

print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[49][1])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#############################################################
#               3 layers CNN Model
############################################################
class LeNet(Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,kernel_size=(5, 5), stride=(2, 2))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv3 = Conv2d(in_channels=50, out_channels=100, kernel_size=(5, 5), stride=(2, 2))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=16900, out_features=500)
        self.relu3 = ReLU()

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


######################################################
#                  Training
######################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

trainSteps = len(train_loader.dataset) // BATCH_SIZE
valSteps = len(valid_loader.dataset) // BATCH_SIZE

print("[INFO] initializing the LeNet model...")
model = LeNet(numChannels=3, classes=13).to(device)

opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

H = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [] }
print("[INFO] training the network...")
startTime = time.time()

for e in range(0, EPOCHS):
    model.train()
    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    for (x, y) in train_loader:
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        loss = lossFn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()
        for (x, y) in valid_loader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            totalValLoss += lossFn(pred, y)
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        model.train()

    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    trainCorrect = trainCorrect / len(train_loader.dataset)
    valCorrect = valCorrect / len(valid_loader.dataset)

    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)

    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))


####################################################
#           Evaluating on Test Set
###################################################
print("[INFO] evaluating network on test set...")
with torch.no_grad():
    model.eval()
    preds = []
    labels = []

    for (x, y) in test_loader:
        x = x.to(device)

        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        labels.extend(y.cpu().numpy())

prec = metrics.precision_score(np.array(labels), np.array(preds), average='weighted')
recall = metrics.recall_score(np.array(labels), np.array(preds), average='weighted')
f1_weighted = metrics.f1_score(np.array(labels), np.array(preds), average='weighted')
f1_macro = metrics.f1_score(np.array(labels), np.array(preds), average='macro')
f1_micro = metrics.f1_score(np.array(labels), np.array(preds), average='micro')

print("Test Precision ", prec)
print("Test Recall ", recall)
print("Test F1-Weighted ", f1_weighted)
print("Test F1-Macro ", f1_macro)
print("Test F1-Micro ", f1_micro)

print("classification report")
print(classification_report( np.array(labels), np.array(preds), target_names=classes))

from sklearn.metrics import balanced_accuracy_score
ba = balanced_accuracy_score(np.array(labels), np.array(preds))
print("Test SKlearn balance accuracy", ba)

from sklearn.metrics import accuracy_score
accry = accuracy_score(np.array(labels), np.array(preds))
print("Test SKlearn accuracy", accry)