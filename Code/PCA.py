# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 08:46:15 2023

@author: Arpit Mittal
"""
import torch
import os
import numpy as np
import math
# importing packages
from sklearn.decomposition import PCA
# In[ ]:
mean_embeddingsPath = "../imagenet_embeddings_resnet/"
mean_embeddings = []
embeddings_label = []
embeddings_folder_label = []
for class_id in os.listdir(mean_embeddingsPath):
        print(class_id)
        embedding_path = mean_embeddingsPath + class_id 
        embeddings = torch.load(embedding_path)
        
        order = np.argsort(np.mean(embeddings,axis=1))
        embeddings_sorted = embeddings[order]
        mean_factor = len(embeddings_sorted)
        truncated_length = math.floor(len(embeddings_sorted)/mean_factor) * mean_factor

        embeddings_sorted_truncated = embeddings_sorted[:truncated_length]
        
        embeddings_mean_at_factor = embeddings_sorted_truncated.reshape(int(truncated_length / mean_factor) , mean_factor , 2048)
        
        embeddings_mean = embeddings_mean_at_factor.mean(axis = 1)
        
        mean_embeddings.extend(embeddings_mean)
        for embed in range(len(embeddings_mean)):
            embeddings_label.append(class_id.split('.')[0].split('_')[1])
            embeddings_folder_label.append(class_id.split('_')[0])
        
mean_embeddings = np.array(mean_embeddings)
embeddings_label = np.array(embeddings_label)
embeddings_folder_label = np.array(embeddings_folder_label)

torch.save(mean_embeddings, '00_mean_embeddings_resnet.pth')
torch.save(embeddings_label, '00_embeddings_label_resnet.pth')
torch.save(embeddings_folder_label, '00_embeddings_folder_label_resnet.pth')