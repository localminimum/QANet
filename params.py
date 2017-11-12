class Params():

    # data
    data_size = -1 # -1 to use all data
    num_epochs = 13
    train_prop = 0.9 # Not implemented atm
    data_dir = "./data/"
    train_dir = data_dir + "trainset/"
    dev_dir = data_dir + "devset/"
    logdir = "./train/train"
    glove_dir = "./glove.840B.300d.txt" # Glove file name (If you want to use your own glove, replace the file name here)
    glove_char = "./glove.840B.300d.char.txt" # Character Glove file name
    coreNLP_dir = "./stanford-corenlp-full-2017-06-09" # Directory to Stanford coreNLP tool

    # Data
    target_dir = "indices.txt"
    q_word_dir = "words_questions.txt"
    q_chars_dir = "chars_questions.txt"
    p_word_dir = "words_context.txt"
    p_chars_dir = "chars_context.txt"

    # Training
    mode = "train" # case-insensitive options: ["train", "test", "debug"]
    LearningRate = 1e-3
    dropout = 0.1 # dropout probability, if None, don't use dropout
    decay = 0.9999 # decay rate of the exponential moving average
    optimizer = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 32 if mode is not "test" else 100 # Size of the mini-batch for training
    save_steps = 50 # Save the model at every 50 steps
    warmup_steps = 1000
    clip = True # clip gradient norm
    norm = 5.0 # global norm
    # NOTE: Change the hyperparameters of your learning algorithm here
    opt_arg = {'adadelta':{'rho': 0.95, 'epsilon':1e-6},
                'adam':{'beta1':0.8, 'beta2':0.999, 'epsilon':1e-7},
                'gradientdescent':{},
                'adagrad':{}}

    # Architecture
    num_heads = 8 # Number of heads in multihead or branched attention
    attention = "multihead" # Which attention to use for multihead, options: ["multihead", "branched"]
    max_p_len = 300 # Maximum number of words in each passage context
    max_q_len = 30 # Maximum number of words in each question context
    max_char_len = 16 # Maximum number of characters in a word
    vocab_size = 91604 # Number of vocabs in glove.840B.300d.txt + 1 for an UNK token
    char_vocab_size = 95 # Number of characters in glove.840B.300d.char.txt + 1 for an UNK character
    emb_size = 300 # Embeddings size for words
    char_emb_size = 200 # Embeddings size for words
    bias = True # Use bias term in attention
    num_units = 128 # Number of units to use throughout the networks
