#!/usr/bin/env python
# coding: utf-8
# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from imageNetDataset import ImageNetDataset
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available
print(f"Using device: {device}")

# In[]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# Define the validation data loader
data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
# In[ ]:
    
root_src = "../../tiered_imagenet_animals/tiered_imagenet_animals/"
root_src = "../../data/"
val_data_src = 'val'
data_src = val_data_src

dataset = ImageNetDataset(root_src, data_src, data_transforms)
#val_data = torchvision.datasets.ImageFolder(root_src + val_data_src, transform=data_transforms)
dataloader = DataLoader(
            dataset,
            batch_size=1, # may need to reduce this depending on your GPU 
            shuffle=False,
        )

# In[ ]:
    
print('----------')
resnet152_model = torchvision.models.resnet152(weights="DEFAULT")
resnet152_model.to(device)
resnet152_model.eval()
correct = 0
total = 0
y_true = []
y_pred = []
print('Validating Resnet Model')
with torch.no_grad():
    for x, y in tqdm(dataloader):
        y_true.extend(y.numpy())
        y_pred.extend(resnet152_model(x.cuda()).argmax(axis=1).cpu().numpy())
        
resnet152_accuracy = accuracy_score(y_true, y_pred)
resnet152_precision = precision_score(y_true, y_pred, average='weighted')
resnet152_recall = recall_score(y_true, y_pred, average='weighted')
resnet152_f1_score = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {resnet152_accuracy:.2f}")
print(f"Precision: {resnet152_precision:.2f}")
print(f"Recall: {resnet152_recall:.2f}")
print(f"F1 score: {resnet152_f1_score:.2f}")

resnet152_accuracy = (100 * correct / total)

# In[ ]:
print('----------')
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0')
efficientnet_model.to(device)
efficientnet_model.eval()
correct = 0
total = 0
y_true = []
y_pred = []
print('Validating EfficientNet Model')
with torch.no_grad():
    for x, y in tqdm(dataloader):
        y_true.extend(y.numpy())
        y_pred.extend(efficientnet_model(x.cuda()).argmax(axis=1).cpu().numpy())
       
efficientnet_b_accuracy = accuracy_score(y_true, y_pred)
efficientnet_b_precision = precision_score(y_true, y_pred, average='weighted')
efficientnet_b_recall = recall_score(y_true, y_pred, average='weighted')
efficientnet_b_f1_score = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {efficientnet_b_accuracy:.2f}")
print(f"Precision: {efficientnet_b_precision:.2f}")
print(f"Recall: {efficientnet_b_recall:.2f}")
print(f"F1 score: {efficientnet_b_f1_score:.2f}")


# In[ ]:

print('----------')
inceptionV3_model = torchvision.models.inception_v3(pretrained=True)
inceptionV3_model.eval()
correct = 0
total = 0
y_true = []
y_pred = []
with torch.no_grad():
    for x, y in tqdm(dataloader):
        y_true.extend(y.numpy())
        y_pred.extend(inceptionV3_model(x.cuda()).argmax(axis=1).cpu().numpy())
       

inception_net_accuracy = accuracy_score(y_true, y_pred)
inception_net_precision = precision_score(y_true, y_pred, average='weighted')
inception_net_recall = recall_score(y_true, y_pred, average='weighted')
inception_net_f1_score = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {inception_net_accuracy:.2f}")
print(f"Precision: {inception_net_precision:.2f}")
print(f"Recall: {inception_net_recall:.2f}")
print(f"F1 score: {inception_net_f1_score:.2f}")


# In[ ]:
print('----------')
# print('Accuracy of the ResNet model on the validation set: %d %%' % (resnet152_accuracy))
# print('Accuracy of the EfficientNet model on the validation set: %d %%' % (efficientnet_accuracy))
# print('Accuracy of the InceptionNet model on the validation set: %d %%' % (inceptionV3_accuracy))
print('----------')
