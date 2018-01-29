# FAST AND ACCURATE READING COMPREHENSION WITHOUT RECURRENT NETWORKS
A Tensorflow implementation of https://openreview.net/pdf?id=B14TlG-RW. Without RNNs the model converges relatively fast compared to [R-net](https://github.com/minsangkim142/R-net). The model converges to devset EM/F1 score of 44/58 within 1 hour of training without dropout in GTX1080 with given parameters in param.py file.

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
$ python process.py --process True --reduce_glove True
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
- [x] Convergence testing
- [ ] Apply dropout every 2 layers
- [ ] Query-to-context attention
- [ ] Data augmentation by paraphrasing

## Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=./
```

## Note
**28/01/18**
The model reaches devset performance of EM/F1=44/58 1 hour into training without dropout. Next goal is to train with dropout every 2 layers.
**04/11/17**
Currently the model is not optimized and there is a memory leak so I strongly suggest only training if your memory is 16GB >. Also I haven't done convergence testing yet. The training time is 5 ~ 6x faster on naive implementation compared to R-net (https://github.com/minsangkim142/R-net).
