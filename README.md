# FAST AND ACCURATE READING COMPREHENSION WITHOUT RECURRENT NETWORKS
This is still a draft as of 04/11/17.

A Tensorflow implementation of https://openreview.net/pdf?id=B14TlG-RW

The dataset used for this task is Stanford Question Answering Dataset (https://rajpurkar.github.io/SQuAD-explorer/). Pretrained GloVe embeddings are used for words (https://nlp.stanford.edu/projects/glove/).

## Requirements
  * Python2.7
  * NumPy
  * nltk
  * tqdm
  * TensorFlow == 1.2

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
