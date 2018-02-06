class Params():
    '''A set of hyper-parameters used for training and testing.
    Automatically saved at the logdir when training starts.'''

    # data
    data_size = -1 # -1 to use all data
    num_epochs = 30
    data_dir = "./data/"
    train_dir = data_dir + "trainset/"
    dev_dir = data_dir + "devset/"
    logdir = "./train/train"
    glove_dir = "./glove.840B.300d.txt" # Glove file name (If you want to use your own glove, replace the file name here)
    glove_char = "./glove.840B.300d.char.txt" # Character Glove file name

    # Data
    target_dir = "indices.txt"
    q_word_dir = "words_questions.txt"
    q_chars_dir = "chars_questions.txt"
    p_word_dir = "words_context.txt"
    p_chars_dir = "chars_context.txt"

    # Training
    mode = "train" # case-insensitive options: ["train", "test", "debug"]
    LearningRate = 1e-3
    l2_norm = 3e-7
    dropout = 0.1 # dropout probability, if None, don't use dropout
    decay = 0.9999 # decay rate of the exponential moving average
    optimizer = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 32 if mode is not "test" else 32 # Size of the mini-batch for training
    dev_batchs = 50
    dev_steps = 1000
    save_steps = 500 # Save the model at every 50 steps
    warmup_steps = 1000
    clip = True # clip gradient norm
    norm = 5.0 # global norm
    # NOTE: Change the hyperparameters of your learning algorithm here
    opt_arg = {'adadelta':{'rho': 0.95, 'epsilon':1e-6},
                'adam':{'beta1':0.8, 'beta2':0.999, 'epsilon':1e-7},
                'gradientdescent':{},
                'adagrad':{}}

    # Architecture
    num_heads = 1 # Number of heads in multihead or branched attention
    #NOTE branched attention is disabled
    attention = "multihead" # Which attention to use for multihead, options: ["multihead", "branched"]
    max_p_len = 400 # Maximum number of words in each passage context
    max_q_len = 30 # Maximum number of words in each question context
    max_char_len = 16 # Maximum number of characters in a word
    char_vocab_size = 95 # Number of characters in glove.840B.300d.char.txt + 1 for an UNK character
    emb_size = 300 # Embeddings size for words
    char_emb_size = 200 # Embeddings size for words
    bias = True # Use bias term in attention
    num_units = 125 # Number of units throughout the networks

    def dump_config(self, dict_):
        line = ["Hyper Parameters for train: " + self.logdir]
        for key in dict_.keys():
            if key not in ["__doc__","__module__"]:
                line.append("{}: {}".format(key, dict_[key]))
        with open(self.logdir + "/config.txt", "w") as f:
            f.write("\n".join(line))
