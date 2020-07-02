# Unsupervised Sentiment Analysis


> Aim: Analyze work and performance through different aspects and factors under various domains based on feedback.

> Impact: help users to analyze work and performance through different aspects and factors under various domains based on feedback.


## Data

The data used is a part of an open-source data available on Kaggle, it can be downloaded from [[Link]](https://www.kaggle.com/fireball684/hackerearthericsson?select=train.csv). The extracted data used is available under the ./data/ folder. It contains the original dataset extracted, pre-processed dataset, and the final dataset used for the current purpose. You can use the data available under ./data/final/ folder for training and testing purposes.  


## Technologies Used:

* Bert Model - Transformer
* Transfer Learning
* K-Means
* K Nearest Neighbor



## Visualization:

* textblob
* seaborn
* matplotlib
* TensorFlow projector
* umap
* PCA
* t-SNE


## Dependencies

* Tensorflow
* Numpy
* Python
* NLTK
* sentence transformers/bert embeddings


## Pre-Processing

The pre-processed data can be found under ./data/final/ folder. 

For reading the dataset and writing it to an array of sentences, the following two codes are used:

- train_read.py
- test_read.py



## Word Embeddings


The Word2Vec Embeddings are used for transforming the sentences to vectors: 

The code used for this purpose is under ./K-Means/BERT+KMeans+TensorFlow_Projector_(PCA_TSNE).ipynb 

However, these embeddings did not prove to be efficient for our purpose. To overcome this, we have introduced Transfer Learning to our model by using Bert as a model.


## BERT

To get the efficient sentence embeddings, Bert is used in our task. Bert can be used in the following two ways:

1) Bert as a service

Install the Bert server and client and start the service. Pre-trained Bert model can be downloaded from [[Link]](https://github.com/hanxiao/bert-as-service)

Then, start the BERT service and use the Client to get Sentence Encodes. The code for the same can be found in these two files:

- Bert_as_service.py
- BERT_as_a_Service.ipynb

2) Install Bert Model

Install the Bert model to get sentence embeddings using pip installer. The dependencies for using this can be found in the following file. It contains a step-wise approach to use Bert Model.

- Bert_model.py





## K-Means

The Clustering algorithm used to make clusters of the bert embeddings is K-Means.
The code for making two separate clusters of positive and negative sentences can be found under ./K-Means/K-Means (2-clusters-+ve and -ve).py

The T-distributed Stochastic Neighbor Embedding is also used for visualizing the embeddings. Following codes contains the code for K-Means Clustering:

- a) 2 Clusters -> ./K-Means/K-Means (2-clusters-+ve and -ve).py
- b) 5 Clusters -> ./K-Means/K-Means (5-clusters).py

These can be changed to any number of clusters (value of K) by changing the parameters in the code above.

Refer: A step-by-step approach for the task is implemented under the ./K-Means/BERT+KMeans+TensorFlow_Projector_(PCA_TSNE).ipynb.



## K Nearest Neighbor (KNN)

A KNN model is also applied on the Bert embeddings to check on the test data.
For initializing the K value, a code for finding the optimal value of K can be found under ./KNN/Initialize_K_for_KNN.py

The code to check on the test data and compute the accuracy metric is available in ./KNN/K_Nearest_Neighbor_KNN.py

Refer: A step-by-step approach for the task is implemented under the ./KNN/Bert+K_Nearest_Neighbour_(KNN).ipynb




## TensorFlow Projector

To plot the n-dimensional embeddings of the data sentences obtained from any model or embeddings (Bert/ Word2Vec), you need to create a DataFile of the embeddings and a MetaData File for assigned a label to every embedding. Then visualization can be done through TensorFlow Projector([[Link]](https://projector.tensorflow.org/)).

The code to generate DataFile and MetaData File of any dataset can be found under ./tensorflow_projector/.

Word2Vec_Embeddings_Clustering - 2_Clusters.py -> Word2Vec Visualization 
Bert_Embeddings_Clustering - 2_Clusters.py -> Bert Visualization  
Bert_Embeddings_Clustering - 5_Clusters.py -> Bert Visualization  
Bert_Embeddings_Clustering - 10_Clusters.py -> Bert Visualization  

The data file and metadata file for the dataset used in our task is already computed and can be found under ./data/final/projector/. 

Refer: A step-by-step approach for the task is implemented under the ./tensorflow_projector/BERT+KMeans+TensorFlow_Projector_(PCA_TSNE).ipynb



## Accuracy Metric

The accuracy of the model: 

```
K-Means (on Bert Embeddings) accuracy: 0.9377
KNN (on Bert Embeddings) accuracy: 0.92735
```



