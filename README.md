# FAST AND ACCURATE READING COMPREHENSION WITHOUT RECURRENT NETWORKS
This is still a draft as of 04/11/17.

A Tensorflow implementation of https://openreview.net/pdf?id=B14TlG-RW

The paper has a few missing information and ambiguity. For example,
1.  They mention max-pooling the character embeddings to obtain word representation. However in the next sentence, they mention using convolution on top of the character embedding (Probably similar implementation to [Char-CNN](https://arxiv.org/pdf/1508.06615.pdf) with [depthwise separable convolution](https://arxiv.org/pdf/1610.02357.pdf) instead of normal convolution). For simplicity we use max-pooling instead of Char-CNN.
2.  They don't mention how they facilitate the difference in input and output dimention for residual connection. Input embeddings have a dimention of 2 * p, and is projected down to 128 before applying layer_norm.
3.  Some of the hyperparameter details are missing, e.g. the number of heads in multi-head attention and so on.

## Dataset
The dataset used for this task is Stanford Question Answering Dataset (https://rajpurkar.github.io/SQuAD-explorer/). Pretrained GloVe embeddings obtained from common crawl with 840B tokens are used for words (https://nlp.stanford.edu/projects/glove/).

## Requirements
  * Python2.7
  * NumPy
  * tqdm
  * TensorFlow (1.2 or higher)
  * spacy

## Downloads and Setup
Preprocessing step is identical to R-net (https://github.com/minsangkim142/R-net). Once you clone this repo, run the following lines from bash **just once** to process the dataset (SQuAD).
```shell
$ pip install -r requirements.txt
$ bash setup.sh
$ python process.py --process True
```

## Training / Testing / Debugging
You can change the hyperparameters from params.py file to fit the model in your GPU. To train the model, run the following line.
```shell
$ python model.py
```
To test or debug your model after training, change mode = "train" from params.py file and run the model.

## TODO's
- [x] Training and testing the model
- [x] Add trilinear function to Context-to-Query attention
- [ ] Convergence testing
- [ ] Apply dropout every 2 layers
- [ ] Data augmentation by paraphrasing

## Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=./
```

## Note
**04/11/17**
Currently the model is not optimized and there is a memory leak so I strongly suggest only training if your memory is 16GB >. Also I haven't done convergence testing yet. The training time is 5 ~ 6x faster on naive implementation compared to R-net (https://github.com/minsangkim142/R-net).
