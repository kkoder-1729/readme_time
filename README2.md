# Aspect-level Supervised Sentiment Analysis (Entity/ Feature-level Sentiment Analysis)

> Aim: Analyze feedbacks by extracting the topics (aspects) and then evaluating the sentiment of each topic. Analyze feedback by evaluating the sentiment of multiple entities present in the feedback. 

> Impact: help users to get the aspect scores of their feedbacks to analyze their weak and strong domain. Useful to generate a performance report of an employee based on feedbacks. Helpful for a user to understand where the effort is missing and where to continue with the same pace based on his/her feedbacks.


## Data

The data used is an open-source data available on SemEval, it can be downloaded from [[Link]](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools). 

The data used is already available under ./dataset/ folder. It contains data of SemEval 14, SemEval 15, and SemEval 16 as data_14, data_15 and data_16 respectively. The document-level datasets can be found under ./dataset/labels/.


## Technologies Used:

* RNN
* Attention Mechanism
* LSTM


## Dependencies

* Tensorflow
* Keras
* Numpy
* Python
* NLTK



## Pre-Training

The pretrained weights used in our task are provided at pretrained_weights/. You can use them directly for initializing aspect-level models.

Or if you want to retrain on another dataset, execute the command below under code_pretrain/:

```
CUDA_VISIBLE_DEVICES="0" python pre_train.py \
--domain labels \
```

The trained model parameters will be saved under pretrained_weights/. You can find more arguments defined in pre_train.py with default values used in our experiments.




## Word Embeddings

The pre-trained Glove vectors (on 840B tokens) are used for initializing word embeddings. They can be downloaded from [[Link]](https://nlp.stanford.edu/projects/glove/). These contain the following 4 glove embeddings. You can download glove embeddings having 840B tokens for our purpose.

- glove.6B.zip
- glove.42B.300d.zip
- glove.840B.300d.zip
- glove.twitter.27B.zip


However, the extracted subset of Glove vectors for the dataset is available under ./glove/ folder, the size of which is much smaller as it is the subset of the original glove embeddings based on the used dataset. 

The code used for word embeddings is embeddings.py



## Training and evaluation

To train aspect-level sentiment classifier, execute the command below under code/:

```
CUDA_VISIBLE_DEVICES="0" python main.py \
--domain $domain \
--alpha 0.1 \
--is-pretrain 1 \
```

where 
$domain in ['data_14', 'data_15', 'data_16'] denotes the corresponding aspect-level domain. 
--alpha denotes the weight of the labels training objective. 
--is-pretrain is set to either 0 or 1, denoting whether to use pretrained weights for initialization. 

You can find more arguments defined in train.py with default values used in our experiments. At the end of each epoch, results on training, validation, and test sets will be printed respectively.

The codes used for training and testing purpose are available in:

- build.py
- attention_layer.py
- adam_opt.py
- main.py


The step-by-step illustration of the whole task is implemented under ./IPYNB FILES/ folder. The steps can be followed in the same way as implemented in .ipynb files. 




## Accuracy Metric

The accuracy of the model: 

```
Validation accuracy: 0.8665
Test accuracy: 0.8192
```


## References
```
- https://towardsdatascience.com/the-mechanics-of-attention-mechanism-f6e9805cca66
- https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f
- https://medium.com/sap-machine-learning-research/effective-attention-modeling-for-aspect-level-sentiment-classification-a49382b4d275
- https://towardsdatascience.com/day-105-of-nlp365-nlp-papers-summary-aspect-level-sentiment-classification-with-3a3539be6ae8
- https://link.springer.com/article/10.1186/s13673-019-0196-3
- https://arxiv.org/abs/1512.01100
- http://www.aclweb.org/anthology/D/D16/D16-1058.pdf
- https://www.aclweb.org/anthology/C18-1096/
- http://arxiv.org/abs/1605.08900
- https://ieeexplore.ieee.org/document/8597286
- https://paperswithcode.com/task/aspect-based-sentiment-analysis
- https://arxiv.org/abs/1806.04346
- https://link.springer.com/chapter/10.1007/978-981-13-1610-4_23
- https://monkeylearn.com/blog/aspect-based-sentiment-analysis/
```