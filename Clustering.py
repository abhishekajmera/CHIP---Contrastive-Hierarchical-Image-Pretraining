# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 08:46:15 2023

@author: Arpit Mittal
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# importing packages
from PIL import Image
import glob
import sklearn.metrics as metrics
import copy
import os

# In[ ]:

def view_cluster(files,level,index):
    plt.figure(figsize = (25,25));
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        for filename in glob.glob("../imagenet/train/" + file +"/*.JPEG"): #assuming gif
            img = Image.open(filename)
            img = np.array(img)
            break
        plt.imshow(img)
        plt.axis('off')
    plt.savefig("clusterImages/"+level+"/"+str(index)+'.png')
    
# In[ ]:
mean_embeddingsPath = "../imagenet_embeddings_resnet/"
mean_embeddings = []
embeddings_label = []
embeddings_folder_label = []
for class_id in os.listdir(mean_embeddingsPath):
        print(class_id)
        embedding_path = mean_embeddingsPath + class_id 
        embeddings = torch.load(embedding_path)
        mean_embeddings.extend(embeddings)
        for embed in embeddings:
            embeddings_label.append(class_id.split('.')[0].split('_')[1])
            embeddings_folder_label.append(class_id.split('_')[0])
# In[ ]:
mean_embeddings = mean_embeddings
embeddings_label = embeddings_folder_label
'''
'''
def optimalK(mean_embeddings,minRange,maxRange,incRange,level):
    # K-means Clustering
    sse = []
    silhouette_avg = []
    kmeans = {}
    list_k = list(range(minRange, maxRange, incRange))
    print("Kmeans - sse")
    for k in list_k:
        print(k)  
        km = KMeans(n_clusters = k, init='k-means++', random_state = 42)
        km.fit(mean_embeddings)
        kmeans[k] = km
        cluster_labels = km.labels_
        sse.append(km.inertia_)
        silhouette_avg.append(metrics.silhouette_score(mean_embeddings, cluster_labels))

    plt.plot(list_k,silhouette_avg,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel("Silhouette score") 
    plt.title('Silhouette analysis For Optimal k')
    plt.show()
    plt.savefig('Silhouette score '+level+'.png')
        
    print("plotting")
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse)
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.savefig('Sum of squared distance '+level+'.png')
    
def kmeansclustering(mean_embeddings,embeddings_label,level,featuresize,inputsize,k):
    strin = '10_'+str(level)+'_'+str(inputsize)+'x'+str(featuresize)+'_resnet_Kmeans_k_'+str(k)
    print("Computing - " + strin)
    km = KMeans(n_clusters = k, init='k-means++', random_state = 42)
    km.fit(mean_embeddings)
    
    groups = {}
    got_folder_set = set()
    got_2_names = set()
    for file, cluster in zip(embeddings_label,km.labels_):
        if cluster not in groups.keys():
            groups[cluster] = set()
            groups[cluster].add(file)
        else:
            groups[cluster].add(file)
        got_folder_set.add(file)
        if len(groups[cluster]) > 1:
            got_2_names.add(cluster)
    print(groups)
    print("Saving - " + strin)
    torch.save(km, '10_allmean_'+str(level)+'_'+str(inputsize)+'x'+str(featuresize)+'_resnet_Kmeans_k_'+str(k)+'.pth')
    torch.save(groups, '10_'+level+'_group.pth')
    print("Done - " + strin)
    return km, groups
# In[ ]:
level_L0 = "level0"
mean_embeddings_L0 = np.array(mean_embeddings)
embeddings_label_L0 = np.array(embeddings_label)
inputsize_L0 = mean_embeddings_L0.shape[0]
featuresize_L0 = mean_embeddings_L0.shape[1]
kvalue_L0 = len(set(embeddings_label_L0))
km_L0,group_L0 = kmeansclustering(mean_embeddings_L0,embeddings_label_L0,level_L0,featuresize_L0,inputsize_L0,kvalue_L0)

# In[ ]:
'''
    
level_L1 = "level1"
mean_embeddings_L1 = torch.load("00_mean_embeddings_resnet.pth")
embeddings_label_L1 = torch.load("00_embeddings_folder_label_resnet.pth")
inputsize_L1 = mean_embeddings_L1.shape[0]
featuresize_L1 = mean_embeddings_L1.shape[1]

# To find optimal K
optimalK(mean_embeddings_L1,100,120,1,level_L1)
kvalue_L1 = 107
km_L1,group_L1 = kmeansclustering(mean_embeddings_L1,embeddings_label_L1,level_L1,featuresize_L1,inputsize_L1,kvalue_L1)

index = 1
for cluster in group_L1.keys():
    index += 1
    view_cluster(group_L1[cluster],level_L1,index)
# In[ ]:
  
level_L2 = "level2"
mean_embeddings_L2 = copy.deepcopy(km_L1.cluster_centers_)
embeddings_label_L2 = copy.deepcopy(km_L1.labels_)
inputsize_L2 = mean_embeddings_L2.shape[0]
featuresize_L2 = mean_embeddings_L2.shape[1]

# To find optimal K
optimalK(mean_embeddings_L1,5,20,1)

kvalue_L2 = 17
km_L2,group_L2 = kmeansclustering(mean_embeddings_L2,embeddings_label_L2,level_L2,featuresize_L2,inputsize_L2,kvalue_L2)
''' 