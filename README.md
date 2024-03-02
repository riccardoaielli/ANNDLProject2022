# Artificial Neural networks and Deep Learning Challenges

## 1) Multiclass classification with CNNs on unbalanced dataset

### Introduction
This repo contains our work regarding the [1st challenge](https://codalab.lisn.upsaclay.fr/competitions/8522#learn_the_details) of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano.
The goal of this challenge is to model a neural network able to classify images of 8 different species of plants training a neural network with a given dataset.

The detailed report of our work can be found [here](Multiclass_classification/Report.pdf).

### Dataset
The dataset contains 3542 images of size 96x96 and divided into the following classes:

    Species1 : 186
    Species2 : 532
    Species3 : 515
    Species4 : 511
    Species5 : 531
    Species6 : 222
    Species7 : 537
    Species8 : 508
    
#### Example of images

![00001](https://user-images.githubusercontent.com/79714834/219965212-f198f79a-fdec-43c9-a029-156d6b26d101.jpg)
![00023](https://user-images.githubusercontent.com/79714834/219965225-12fae437-2945-47e7-80dd-99b0a4fe192d.jpg)
![00022](https://user-images.githubusercontent.com/79714834/219965233-e48733bd-eac5-4c81-ba73-013f2399b4e1.jpg)
![00023](https://user-images.githubusercontent.com/79714834/219965288-225c3b42-6bc7-451d-944a-6f813e7ec37d.jpg)
![00036](https://user-images.githubusercontent.com/79714834/219965387-68826d47-53eb-47f9-b265-f06bd7acbf58.jpg)
![00039](https://user-images.githubusercontent.com/79714834/219965724-b5996c5f-fec5-455d-a6e6-3c76e7c1b5ce.jpg)
![00232](https://user-images.githubusercontent.com/79714834/219965813-c44c3626-cc11-4f09-b353-e437bcac34c1.jpg)
![00049](https://user-images.githubusercontent.com/79714834/219965835-696b156e-e5ff-493b-a91b-241d7df0c8c9.jpg)

### Improving data quality

We adopted multiple techniques to improve the quality of the given dataset

- Data augmentation 
- Removing low quality images (e.g. images with external objects)
- Upsampling of underrepresented classes

### Transfer learning

Our best scoring models make use of transfer learning.

We applied the classic Transfer Learning workflow:
1. Using the pretrained network as a feature extractor by freezing its layers
2. Fine tuning by unfreezing partly or completely the layers of the pretrained network

Our best scoring model uses ConvNeXtBase, pretrained on the ImageNet datasets.

### Results

Our best model reached an accuracy of 0.9113, with an F1 score of the most scarce and problematic class (species 1) of 0.7801.

F1-score table on the private test set:
||Species 1|Species 2|Species 3|Species 4|Species 5|Species 6|Species 7|Species 8|
|---|---|---|---|---|---|---|---|---|
|F1-score|0.7801|0.8981|0.9684|0.9326|0.9486|0.9477|0.9666|0.8388|

## 2) Multivariate time series classification with deep learning techniques

### Introduction

This repo contains our work regarding the [2nd challenge](https://codalab.lisn.upsaclay.fr/competitions/9056#learn_the_details) of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano. The goal of this challenge is to model a neural network able to classify time series characterized by sequences of 36 samples, 6 features and 12 possible labels.

The detailed report of our work can be found [here](Multivariate_time_series_classification/Report.pdf).

### Data preprocessing

The source of the dataset is unknown, thus no domain knowledge could be applied. 
We employed different techniques to improve our result by exploiting at most our data:
- **Data Standardization**: we used RobustSCaler, a scaler robust to outliers, to improve stability and performance of our model

- **Time Series Augmentation**: we used the library [tsaug](https://tsaug.readthedocs.io/en/stable/) and experimented with various types of augmentation

- **Oversampling and Undersampling**: Our dataset was imbalanced with a class having 777 samples and another one only 34. To make up for this imbalance we applied oversampling and undersampling techniques, with oversampling giving the best result.

- **Sliding Windows**: Under the assumption that the training dataset contained chronologically ordered samples, as a form of further augmentation, we developed an algorithm to build sliding window with random sampling of validation and test set

### Models

We experimented with various models, including LSTM, BiLSTM and 1D Convolution.

Additionally, we also tried some ensembles, the model that performed best was a new architecture based on RNN and GRU layers giving us an improvement of 4\% of accuracy.



### Results

Our best model reached an accuracy of 0.7338 and we were able to improve the F1 score on the most problematic classes, scoring 9th best in class 7 among more than 500 models.
