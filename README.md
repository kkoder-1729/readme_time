Projector images
AT-LSTM


# Sentence-level Supervised Sentiment Analysis

> Aim: Analyze feedbacks by finding the polarity of the text.

> Impact: help users to classify the feedback provided into the +ve /-ve category with its polarity score to analyze their performance.


## Data

The data used is a part of an open-source data available on Kaggle, it can be downloaded from [[Link]](https://www.kaggle.com/fireball684/hackerearthericsson?select=train.csv). The extracted data used is available under the ./data/ folder. It contains the original dataset extracted, pre-processed dataset, and the final dataset used for the current purpose. You can use the data available under ./data/final/ folder for training and testing purposes.  


## Technologies Used:

* Bi-RNN
* Attention Mechanism
* LSTM



## Visualization:

* textblob
* wordcloud
* seaborn
* matplotlib


## Dependencies

* Tensorflow
* Numpy
* Python
* NLTK
* multiprocessing


## Pre-Processing

The pre-processed data can be found under the ./data/processed/ folder.
The modules used for preprocessing used are:

- init.py
- utils.py
- preprocess_text.py
- preprocess_csv.py

The MetaData (a set of data that describes and gives information about other data) of the dataset is found using the following modules: 

- metadata.py

It is used to initialize the parameters to build the model. 

- convert.py



## Word Embeddings

The word embeddings used is wiki-news-300d-1M.vec which can be downloaded from [[Link]](https://fasttext.cc/docs/en/english-vectors.html)

Other word embeddings can also be used which can vary in model metrics. They can be downloaded from [[Link]](https://nlp.stanford.edu/projects/glove/). These contains the following 4 glove embeddings.

- glove.6B.zip
- glove.42B.300d.zip
- glove.840B.300d.zip
- glove.twitter.27B.zip

The code used for word embeddings is word_embeddings.py



## Training and Evaluation

The codes used for training and testing purpose are available in:

- bi-rnn.py
- attention.py
- train.py
- metric.py

The step-by-step illustration of the whole task is implemented under ./IPYNB FILES/ folder. The steps can be followed in the same way as implemented in .ipynb files. 




## Accuracy Metric

The accuracy of the model: 

```
Train accuracy: 0.9541
Test accuracy: 0.9388
```

