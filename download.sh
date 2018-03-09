#!/usr/bin/env bash

# Download SQuAD
SQUAD_DIR=~/data/squad
mkdir -p $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json

# Download GloVe
GLOVE_DIR=~/data/glove
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

# Download Glove Character Embedding
# wget https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt -O $GLOVE_DIR/glove.840B.300d-char.txt

# Download fasttext
# FASTTEXT_DIR=~/data/fasttext
# mkdir -p $FASTTEXT_DIR
# wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip -O $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip
# unzip $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip -d $FASTTEXT_DIR

# Download Spacy language models
python3 -m spacy download en
