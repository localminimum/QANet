mkdir data
wget -P data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -P data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
wget -O glove.zip https://www.dropbox.com/s/4r8c888myxqa9hh/glove.zip?dl=0
unzip glove.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
rm glove.zip
