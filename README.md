# FAST AND ACCURATE READING COMPREHENSION WITHOUT RECURRENT NETWORKS
A Tensorflow implementation of Google's [Fast Reading Comprehension](https://openreview.net/pdf?id=B14TlG-RW) from [ICLR2018](https://openreview.net/forum?id=B14TlG-RW).
Training and preprocessing pipeline has been adopted from [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net). Demo mode needs to be reimplemented. If you are here for the demo please use "dev" branch. The model reaches EM/F1 = 66/75 in 30k steps.

Due to memory issue, a single head dot-product attention is used as opposed to 8 heads multi-head attention as mentioned in the original paper. Also hidden size is reduced to 96 from 128 due to memory problems in GTX1080. (8GB GPU memory is insufficient. If you have a 12GB memory GPU please share your results with us.)

![Alt text](/../master/screenshots/figure.png?raw=true "Network Outline")

## Dataset
The dataset used for this task is [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/).
Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens are used for words.

## Requirements
  * Python2.7
  * NumPy
  * tqdm
  * TensorFlow (1.2 or higher)
  * spacy

## Usage

To download and preprocess the data, run

```bash
# download SQuAD and Glove
sh download.sh
# preprocess the data
python config.py --mode prepro
```

Just like [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net), hyper parameters are stored in config.py. To debug/train/test the model, run

```bash
python evaluate-v1.1.py ~/data/squad/dev-v1.1.json log/answer/answer.json
```

The default directory for tensorboard log file is `log/event`

## Detailed Implementaion

  * The model adopts character level convolution - max pooling - highway network for input representations similar to [this paper by Yoon Kim](https://arxiv.org/pdf/1508.06615.pdf).
  * Encoder consists of positional encoding - depthwise separable convolution - self attention - feed forward structure with layer norm in between.
  * Stochastic depth dropout is used to drop the residual connection with respect to increasing depth of the network as this paper heavily relies on residual connection.
  * Context-to-Query attention is used but Query-to-Context attention is not implemented as it is reported not to improve much on the performance.
  * Learning rate increases from 0.0 to 0.001 in first 1000 steps in inverse exponential scale and fixed to 0.001 from 1000 steps.
  * For regularization, dropout of 0.1 is used every 2 sub-layers and 2 blocks.
  * During prediction, this model uses shadow variables maintained by exponential moving average of all global variables.
  * [Taken from R-Net](https://github.com/HKUST-KnowComp/R-Net): To address efficiency issue, this implementation uses bucketing method (contributed by xiongyifan) and CudnnGRU. Due to a known bug [#13254](https://github.com/tensorflow/tensorflow/issues/13254) in Tensorflow, the weights of CudnnGRU may not be properly restored. Check the test score if you want to use it for prediction. The bucketing method can speedup the training, but will lower the F1 score by 0.3%.


## TODO's
- [x] Add trilinear function to Context-to-Query attention
- [x] Apply dropouts + stochastic depth dropout
- [ ] Realtime Demo
- [ ] Query-to-context attention
- [ ] Data augmentation by paraphrasing

## Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=./
```
