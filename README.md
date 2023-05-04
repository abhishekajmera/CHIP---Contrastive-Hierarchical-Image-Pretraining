# CHIP
## The root directory consists of 24 items. Each of these are either a script to run each phase or are the model's weights generated at the end of each phase.
## The files '00_mean_embeddings_*.pth' are the mean embeddings generated using a regular pretrained ResNet-152 model to be used for the clustering in Phase-1 of the code.
## The files '10_allmean_level*_*x2048_resnet_Kmeans_k_*.pth' are the weights for phase 2 by clustering the 366 animal classes in ImageNet to the 366, 88 or 8 clusters for Levels 0, 1 and 2.
## The files '10_level0_group.pth' are the groups of images for each level 0, 1 and 2.
## The script 'BaseLineModels.py' is a pre evaluation comparison of the best pretrained models to use for our task. The models compared were ResNet, EfficientNet, InceptionNet, ViT and SWIN. Out of these a decision was made to use ResNet-152 for our task.
## The folder 'Cluster Centroid Embeddings' consists of the centroid of the embeddings of all images in a particular cluster. Each different file is for a different level.
## The file 'Clustering.py' is the script for Phase-1 of our architecture where the appropriate clusters are found for each level, evaluated using Silhouette analysis.
## The folder 'dataframes' consists of dataframes conveniently storing the information for mean embeddings for each level and each class pertaining to the cluster of that level.
## The folder 'Embeddings Generated' consists of the embeddings generated on the unseen classes for each level.
## The script 'generate_embeddings.ipynb' is the script to generate the embeddings for each image using a fine tuned ResNet encoder using the weights stored in the mean embeddings filed above.
## The file 'imagenet_class_index.json' maps class names appropriately between different dataset sources.
## The file 'imageNetDataset.py' is used to consolidate imagenet images into a nice PyTorch Dataloader
## The file 'mapping.pth' contains the mapping information corresponding to each image.
## The file 'PCA.py' is used to reduce dimensions for the embeddings in the first phase to make clustering faster.
## The file 'Phase 3 - ResNet.ipynb' is the script for Phase 3 of our model for the unseen classes.
## The file 'Scaled.ipynb' is used to scale and transform the unseen classes in a format that is similar in the feature space to the seen classes embeddings
## The file 'Phase2_ResNet_OneShot.py' is the main script for running Phase 2 of the model conducting one shot training for the seen training classes.
## The file 'Phase0_CheckCLusters.py' is the script for checking for possible clusters initially on all images in the dataset
