from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import time
from sklearn.metrics import accuracy_score

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

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

train_image_paths, valid_image_paths = train_image_paths[:int(0.8 * len(train_image_paths))], train_image_paths[
                                                                                              int(0.8 * len(
                                                                                                  train_image_paths)):]

test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))
print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths),
                                                             len(test_image_paths)))


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
        self.transform = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),
                                       ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]

        img = Image.open(image_filepath).convert('RGB')
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
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
valid_dataset = CovidImagesDataset(valid_image_paths)
test_dataset = CovidImagesDataset(test_image_paths)
print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[49][1])


# #######################################################
# #                  Visualize Dataset
# #         Images are plotted after augmentation
# #######################################################
#
# def visualize_augmentations(dataset, idx=0, samples=10, cols=5, random_img=False):
#     dataset = dataset
#     # we remove the normalize and tensor conversion from our augmentation pipeline
#    # dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
#     rows = samples // cols
#
#     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
#     for i in range(samples):
#         if random_img:
#             idx = np.random.randint(1, len(train_image_paths))
#         image, lab = dataset[idx]
#         ax.ravel()[i].imshow(image)
#         ax.ravel()[i].set_axis_off()
#         ax.ravel()[i].set_title(idx_to_class[lab])
#     plt.tight_layout(pad=1)
#     plt.show()
# visualize_augmentations(train_dataset, np.random.randint(1, len(train_image_paths)), random_img=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

 ############################################
          #     RESNET50 model used
 ############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 13),
                         nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

epochs = 10
running_loss = 0
val_running_loss = 0
print_every = 230
train_losses, test_losses = [], []
start = time.time()

for epoch in range(epochs):
    preds = []
    labs = []
    steps = 0
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        labels = labels.cpu().numpy()
        labs.extend(labels)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        for k in range(len(top_class)):
             preds.extend(top_class.data[k].cpu().numpy())
    train_acc = accuracy_score(np.array(labs), np.array(preds))
    print("[INFO] EPOCH: {}/{}".format(epoch, epochs))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(running_loss / steps, train_acc))
    running_loss = 0

    val_preds = []
    val_labs = []
    val_steps = 0
    for inputs, labels in valid_loader:
        val_steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        val_running_loss += loss.item()
        labels = labels.cpu().numpy()
        val_labs.extend(labels)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        for k in range(len(top_class)):
            val_preds.extend(top_class.data[k].cpu().numpy())
    val_acc = accuracy_score(np.array(val_labs), np.array(val_preds))
    print("Validation loss: {:.6f}, Validation accuracy: {:.4f}".format(val_running_loss / val_steps, val_acc))
    val_running_loss = 0

end = time.time()
print("training time", str(end-start))


##############################################################################
#              Evaluate on Test set
##############################################################################
with torch.no_grad():
    model.eval()
    test_preds = []
    test_labels = []
    tmp =0

    for (x, y) in test_loader:
        tmp +=1
        y= y.to(device)
        y = y.cpu().numpy()
        test_labels.extend(y)

        x = x.to(device)
        logps = model.forward(x)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        for k in range(len(top_class)):
            test_preds.extend(top_class.data[k].cpu().numpy())

prec = metrics.precision_score(np.array(test_labels), np.array(test_preds), average='weighted')
recall = metrics.recall_score(np.array(test_labels), np.array(test_preds), average='weighted')
f1_weighted = metrics.f1_score(np.array(test_labels), np.array(test_preds), average='weighted')
f1_macro = metrics.f1_score(np.array(test_labels), np.array(test_preds), average='macro')
f1_micro = metrics.f1_score(np.array(test_labels), np.array(test_preds), average='micro')

print("Test Precision ", prec)
print("Test Recall ", recall)
print("Test F1-Weighted ", f1_weighted)
print("Test F1-Macro ", f1_macro)
print("Test F1-Micro ", f1_micro)

from sklearn.metrics import classification_report
print("classification report")
print(classification_report( np.array(test_labels), np.array(test_preds), target_names=classes))

from sklearn.metrics import balanced_accuracy_score
ba = balanced_accuracy_score(np.array(test_labels), np.array(test_preds))
print("Test SKlearn balance accuracy", ba)

from sklearn.metrics import accuracy_score
accry = accuracy_score(np.array(test_labels), np.array(test_preds))
print("Test SKlearn accuracy", accry)