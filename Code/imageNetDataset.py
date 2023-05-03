#!/usr/bin/env python
# coding: utf-8
# In[ ]:


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os
from torch.utils.data import Dataset
import json
import glob

train_set = set()


class ImageNetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        samples_dir = os.path.join(root, split)
        for entry in os.listdir(samples_dir):
                syn_id = entry
                if entry in train_set:
                    continue
                
                train_set.add(entry)
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                for filename in glob.glob(sample_path+"/*.jpg"): #assuming gif
                    im=Image.open(filename)
                    self.samples.append(self.transform(im))
                    self.targets.append(target)
                    
                break
                    
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            return self.samples[idx], self.targets[idx]
        
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
print('----------')
# Load pre-trained EfficientNet model
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0')

# Remove final classification layer
efficientnet_model.classifier = torch.nn.Identity()
efficientnet_model.to(device)
efficientnet_model.eval()
embeddings = []
embeddings_map = {}
    
root_src = "../imagenet_dummy/"
train_data_src = 'train'

samples_dir = os.path.join(root_src, train_data_src)
num_class = os.listdir(samples_dir)
# In[ ]:

while True:
    if len(train_set) == len(num_class):
        break
    
    train_dataset = ImageNetDataset(root_src, train_data_src, data_transforms)
    train_dataloader = DataLoader(
                train_dataset,
                batch_size=16, # may need to reduce this depending on your GPU 
                shuffle=False
            )
    
    with torch.no_grad():
        for x, y in tqdm(train_dataloader):
            activations = efficientnet_model(x.cuda())
            embeddings.append(torch.nn.functional.normalize(activations.flatten(start_dim=1)))
            if y not in embeddings_map:
                embeddings_map[y] = []
            embeddings_map[y].append(torch.nn.functional.normalize(activations.flatten(start_dim=1)))
torch.save(embeddings, 'imagenet_embeddings.pth')
torch.save(embeddings_map, 'imagenet_embeddings_map.pth')

print("Embeddings Done")
print(len(embeddings))