# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, get_dev
from params import Params
from layers import *
from evaluate import *
import numpy as np
import cPickle as pickle
from process import *
from demo import Demo

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
            "adam":tf.train.AdamOptimizer,
            "gradientdescent":tf.train.GradientDescentOptimizer,
            "adagrad":tf.train.AdagradOptimizer}
initializer = tf.contrib.layers.xavier_initializer

class Model(object):
    def __init__(self,is_training = True, vocab_size = 100000, demo = False):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if demo:
                self.demo_inputs()
            else:
                self.data, self.num_batch = get_batch(is_training = is_training)
                (self.passage_w,
                self.question_w,
                self.passage_c,
                self.question_c,
                self.indices) = self.data

            self.passage_mask = tf.cast(1 - tf.cast(tf.equal(self.passage_w,1), tf.float32), tf.bool)
            self.question_mask = tf.cast(1 - tf.cast(tf.equal(self.question_w,1), tf.float32), tf.bool)
            self.passage_len = tf.reduce_sum(tf.cast(self.passage_mask, tf.int32), axis=1)
            self.question_len = tf.reduce_sum(tf.cast(self.question_mask, tf.int32), axis=1)

            self.encode_ids()
            self.embedding_encoder()
            self.context_to_query()
            self.model_encoder()
            self.output_layer()

            if Params.decay:
                self.apply_ema()

            if is_training:
                self.loss_function()
                self.summary()
                self.init_op = tf.global_variables_initializer()

            total_params()

    def demo_inputs(self):
        self.passage_w = tf.placeholder(tf.int32,
                                [1, Params.max_p_len,],"passage_w")
        self.question_w = tf.placeholder(tf.int32,
                                [1, Params.max_q_len,],"passage_q")
        self.passage_c = tf.placeholder(tf.int32,
                                [1, Params.max_p_len,Params.max_char_len],"passage_pc")
        self.question_c = tf.placeholder(tf.int32,
                                [1, Params.max_q_len,Params.max_char_len],"passage_qc")
        self.indices = tf.placeholder(tf.int32,
                                [1, 2],"indices")
        self.data = (self.passage_w,
                    self.question_w,
                    self.passage_c,
                    self.question_c)

    def encode_ids(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            self.unknown = tf.get_variable("unknown_word", (1, Params.emb_size), dtype = tf.float32, initializer = initializer())
            with tf.device('/cpu:0'):
                self.char_embeddings = tf.get_variable("char_embeddings", (Params.char_vocab_size+1, Params.char_emb_size), dtype = tf.float32, initializer = initializer())
                self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
                self.word_embeddings_placeholder = tf.placeholder(tf.float32,[self.vocab_size, Params.emb_size],"word_embeddings_placeholder")
                self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)
                self.word_embeddings = tf.concat([self.unknown, self.word_embeddings], axis = 0)

            # Embed the question and passage information for word and character tokens
            self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
                                            self.passage_c,
                                            word_embeddings = self.word_embeddings,
                                            char_embeddings = self.char_embeddings)
            self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
                                            self.question_c,
                                            word_embeddings = self.word_embeddings,
                                            char_embeddings = self.char_embeddings)

            self.passage_char_encoded = depthwise_separable_convolution(self.passage_char_encoded,
                kernel_size = (1, 5), num_filters = Params.char_emb_size, scope = "depthwise_char_conv",
                is_training = self.is_training, reuse = None)
            self.question_char_encoded = depthwise_separable_convolution(self.question_char_encoded,
                kernel_size = (1, 5), num_filters = Params.char_emb_size, scope = "depthwise_char_conv",
                is_training = self.is_training, reuse = True)

            self.passage_char_encoded = tf.reduce_max(self.passage_char_encoded, axis = 2)
            self.question_char_encoded = tf.reduce_max(self.question_char_encoded, axis = 2)

            self.passage_word_encoded = tf.nn.dropout(self.passage_word_encoded, 1.0 - self.dropout)
            self.question_word_encoded = tf.nn.dropout(self.question_word_encoded, 1.0 - self.dropout)
            self.passage_char_encoded = tf.nn.dropout(self.passage_char_encoded, 1.0 - 0.5 * self.dropout)
            self.question_char_encoded = tf.nn.dropout(self.question_char_encoded, 1.0 - 0.5 * self.dropout)

            self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded), axis = -1)
            self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded), axis = -1)

            # self.passage_encoding = highway(self.passage_encoding, 128, project = True, scope = "highway", reuse = None)
            # self.question_encoding = highway(self.question_encoding, 128, project = True, scope = "highway", reuse = True)
            self.passage_encoding = tf.nn.dropout(highway(self.passage_encoding, scope = "highway", reuse = None), 1.0 - self.dropout)
            self.question_encoding = tf.nn.dropout(highway(self.question_encoding, scope = "highway", reuse = True), 1.0 - self.dropout)

    def embedding_encoder(self):
        with tf.variable_scope("Embedding_Encoder_Layer"):
            self.passage_context = residual_block(self.passage_encoding,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.passage_mask,
                input_projection = True,
                seq_len = self.passage_len,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout)
            self.question_context = residual_block(self.question_encoding,
                                                  num_blocks = 1,
                                                  num_conv_layers = 4,
                                                  kernel_size = 7,
                                                  input_projection = True,
                                                  seq_len = self.question_len,
                                                  scope = "Encoder_Residual_Block",
                                                  reuse = True,
                                                  bias = False,
                                                  dropout = self.dropout)
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.question_mask,
                input_projection = True,
                seq_len = self.question_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False, # Cannot use bias due to shape mismatch in self attention (300 vs 30)
                dropout = self.dropout)

    def context_to_query(self):
        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            P = tf.tile(tf.expand_dims(self.passage_context,2),[1,1,Params.max_q_len,1])
            Q = tf.tile(tf.expand_dims(self.question_context,1),[1,Params.max_p_len,1,1])
            S = trilinear([P, Q, P*Q], input_keep_prob = 1.0 - self.dropout)
            mask = tf.expand_dims(self.question_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, self.question_len, mask = mask))
            self.c2q_attention = tf.matmul(S_, self.question_context)

    def model_encoder(self):
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat([self.passage_context, self.c2q_attention, self.passage_context * self.c2q_attention], axis = -1)
            self.encoder_outputs = [conv(inputs, Params.num_units, name = "input_projection")]
            for i in range(3):
                if i % 2 == 0: # dropout every 2 blocks
                    self.encoder_outputs[i] = tf.nn.dropout(self.encoder_outputs[i], 1.0 - self.dropout)
                self.encoder_outputs.append(
                                            residual_block(self.encoder_outputs[i],
                                                           num_blocks = 7,
                                                           num_conv_layers = 2,
                                                           kernel_size = 5,
                                                           seq_len = self.passage_len,
                                                           scope = "Model_Encoder",
                                                           reuse = True if i > 0 else None,
                                                           dropout = self.dropout)
                                            )
                if i in [0,2]:
                    residual_block(self.encoder_outputs[i],
                       num_blocks = 7,
                       num_conv_layers = 2,
                       kernel_size = 5,
                       mask = self.passage_mask,
                       seq_len = self.passage_len,
                       scope = "Model_Encoder",
                       reuse = True if i > 0 else None,
                       dropout = self.dropout)
                    )

    def output_layer(self):
        with tf.variable_scope("Output_Layer"):
            self.start_logits = tf.squeeze(conv(tf.concat([self.encoder_outputs[1], self.encoder_outputs[2]],axis = -1),1, bias = False, name = "start_pointer"),-1)
            self.end_logits = tf.squeeze(conv(tf.concat([self.encoder_outputs[1], self.encoder_outputs[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
            self.logits = [mask_logits(self.start_logits, self.passage_len, mask = self.passage_mask),
                           mask_logits(self.end_logits, self.passage_len, mask = self.passage_mask)]

            self.logit_1, self.logit_2 = [tf.nn.softmax(l) for l in self.logits]
            self.logit_1 = tf.expand_dims(self.logit_1, 2)
            self.dp = tf.matmul(self.logit_1, tf.expand_dims(self.logit_2,1))
            self.dp = tf.matrix_band_part(self.dp, 0, 15)
            self.output_index_1 = tf.argmax(tf.reduce_max(self.dp, axis = 2), -1)
            self.output_index_2 = tf.argmax(tf.reduce_max(self.dp, axis = 1), -1)
            self.output_index = tf.stack([self.output_index_1, self.output_index_2], axis = 1)

    def apply_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(Params.decay)
        self.shadow_vars = []
        self.global_vars = []
        for var in tf.global_variables():
            v = self.var_ema.average(var)
            if v:
                self.shadow_vars.append(v)
                self.global_vars.append(var)
        self.assign_vars = []
        for g,v in zip(self.global_vars, self.shadow_vars):
            self.assign_vars.append(tf.assign(g,v))

    def loss_function(self):
        with tf.variable_scope("loss"):
            shapes = self.passage_w.shape
            self.indices_prob = [tf.squeeze(i, 1) for i in tf.split(tf.one_hot(self.indices, shapes[1]), 2, axis = 1)]
            self.mean_losses = [tf.nn.softmax_cross_entropy_with_logits(logits = l, labels = i) for l,i in zip(self.logits, self.indices_prob)]
            self.mean_loss = tf.reduce_mean(sum(self.mean_losses))

            # apply l2 regularization
            if Params.l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                self.mean_loss += l2_loss

            # apply ema
            if Params.decay is not None:
                ema_op = self.var_ema.apply(tf.trainable_variables())
                with tf.control_dependencies([ema_op]):
                    self.mean_loss = tf.identity(self.mean_loss)

            # learning rate warmup scheme
            self.warmup_scheme = tf.minimum(Params.LearningRate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
            # self.warmup_scheme = tf.minimum(1. / 30 * (tf.cast(self.global_step,tf.float32)**-0.5), 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
            # self.warmup_scheme = (Params.num_units ** -0.5) * tf.minimum((tf.cast(self.global_step,tf.float32)**-0.5), tf.cast(self.global_step, tf.float32)*(Params.warmup_steps ** -1.5))
            self.optimizer = optimizer_factory[Params.optimizer](learning_rate = self.warmup_scheme, **Params.opt_arg[Params.optimizer])

            # gradient clipping by norm
            if Params.clip:
                gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step = self.global_step)
            else:
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step = self.global_step)

    def summary(self):
        self.F1 = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="F1")
        self.F1_placeholder = tf.placeholder(tf.float32, shape = (), name = "F1_placeholder")
        self.EM = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="EM")
        self.EM_placeholder = tf.placeholder(tf.float32, shape = (), name = "EM_placeholder")
        self.dev_loss = tf.Variable(tf.constant(10.0, shape=(), dtype = tf.float32),trainable=False, name="dev_loss")
        self.dev_loss_placeholder = tf.placeholder(tf.float32, shape = (), name = "dev_loss")
        self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder),tf.assign(self.dev_loss, self.dev_loss_placeholder))
        tf.summary.scalar('loss_training', self.mean_loss)
        tf.summary.scalar('loss_dev', self.dev_loss)
        tf.summary.scalar("F1_Score",self.F1)
        tf.summary.scalar("Exact_Match",self.EM)
        tf.summary.scalar('learning_rate', self.warmup_scheme)
        self.merged = tf.summary.merge_all()

def debug():
    dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    vocab_size = dict_.vocab_size
    model = Model(is_training = True, vocab_size = vocab_size)
    print("Built model")

def test():
    dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    vocab_size = dict_.vocab_size
    model = Model(is_training = False, vocab_size = vocab_size); print("Built model")
    with model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor()
        with sv.managed_session(config = config) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
            gs = sess.run(model.global_step)
            if Params.decay is not None and gs > 25000:
                shadow_vars = sess.run(model.shadow_vars)
                sess.run(model.assign_vars, {a:b for a,b in zip(model.shadow_vars, shadow_vars)})
            EM, F1 = 0.0, 0.0
            for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                index, ground_truth, passage = sess.run([model.output_index, model.indices, model.passage_w])
                for batch in range(Params.batch_size):
                    f1, em = f1_and_EM(index[batch], ground_truth[batch], passage[batch], dict_)
                    F1 += f1
                    EM += em
            F1 /= float(model.num_batch * Params.batch_size)
            EM /= float(model.num_batch * Params.batch_size)
            print("Exact_match: {}\nF1_score: {}".format(EM,F1))

def demo():
    dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    vocab_size = dict_.vocab_size
    model = Model(is_training = False, vocab_size = vocab_size, demo = True); print("Built model")
    demo_run = Demo(model = model)

def main():
    dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    vocab_size = dict_.vocab_size
    model = Model(is_training = True, vocab_size = vocab_size); print("Built model")
    init = False
    devdata, dev_ind = get_dev()
    if not os.path.isfile(os.path.join(Params.logdir,"checkpoint")):
        init = True
        glove = np.memmap(Params.data_dir + "glove.np", dtype = np.float32, mode = "r")
        glove = np.reshape(glove,(vocab_size,Params.emb_size))
    with model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=Params.logdir,
                        save_model_secs=0,
                        global_step = model.global_step,
                        init_op = model.init_op)
        with sv.managed_session(config = config) as sess:
            if init: sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove})
            pr = Params()
            pr.dump_config(Params.__dict__)
            for epoch in range(1, Params.num_epochs+1):
                train_loss = []
                if sv.should_stop(): break
                for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                    _, loss = sess.run([model.train_op, model.mean_loss],
                                        feed_dict={model.dropout: Params.dropout if Params.dropout is not None else 0.0})
                    train_loss.append(loss)
                    if step % Params.save_steps == 0:
                        gs = sess.run(model.global_step)
                        sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
                    if step % Params.dev_steps == 0:
                        EM_ = []
                        F1_ = []
                        dev = []
                        for i in range(Params.dev_batchs):
                            sample = np.random.choice(dev_ind, Params.batch_size)
                            feed_dict = {data: devdata[i][sample] for i,data in enumerate(model.data)}
                            index, dev_loss = sess.run([model.output_index, model.mean_loss], feed_dict = feed_dict)
                            F1, EM = 0.0, 0.0
                            for batch in range(Params.batch_size):
                                f1, em = f1_and_EM(index[batch], devdata[-1][sample][batch], devdata[0][sample][batch], dict_)
                                F1 += f1
                                EM += em
                            F1 /= float(Params.batch_size)
                            EM /= float(Params.batch_size)
                            EM_.append(EM)
                            F1_.append(F1)
                            dev.append(dev_loss)
                        EM_ = np.mean(EM_)
                        F1_ = np.mean(F1_)
                        dev = np.mean(dev)
                        sess.run(model.metric_assign,{model.F1_placeholder: F1_, model.EM_placeholder: EM_, model.dev_loss_placeholder: dev})
                        print("\nTrain_loss: {}\nDev_loss: {}\nDev_Exact_match: {}\nDev_F1_score: {}".format(np.mean(train_loss),dev,EM_,F1_))
                        train_loss = []

if __name__ == '__main__':
    if Params.mode.lower() == "debug":
        print("Debugging...")
        debug()
    if Params.mode.lower() == "demo":
        print("Running Interactive Demo Session...")
        demo()
    elif Params.mode.lower() == "test":
        print("Testing on dev set...")
        test()
    elif Params.mode.lower() == "train":
        print("Training...")
        main()
    else:
        print("Invalid mode.")
