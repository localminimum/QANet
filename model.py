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

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
            "adam":tf.train.AdamOptimizer,
            "gradientdescent":tf.train.GradientDescentOptimizer,
            "adagrad":tf.train.AdagradOptimizer}
initializer = tf.contrib.layers.xavier_initializer

class Model(object):
    def __init__(self,is_training = True):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.data, self.num_batch = get_batch(is_training = is_training)
            (self.passage_w,
            self.question_w,
            self.passage_c,
            self.question_c,
            self.passage_w_len_,
            self.question_w_len_,
            self.indices) = self.data

            self.passage_len = tf.squeeze(self.passage_w_len_)
            self.question_len = tf.squeeze(self.question_w_len_)

            self.encode_ids()
            self.embedding_encoder()
            self.context_to_query()
            self.model_encoder()
            self.output_layer()

            if is_training:
                self.loss_function()
                self.summary()
                self.init_op = tf.global_variables_initializer()
            total_params()

    def encode_ids(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            with tf.device('/cpu:0'):
                self.char_embeddings = tf.get_variable("char_embeddings", (Params.char_vocab_size+1, Params.char_emb_size), dtype = tf.float32, initializer = initializer())
                self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
                self.word_embeddings_placeholder = tf.placeholder(tf.float32,[Params.vocab_size, Params.emb_size],"word_embeddings_placeholder")
                self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)

            # Embed the question and passage information for word and character tokens
            self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
                                            self.passage_c,
                                            word_embeddings = self.word_embeddings,
                                            char_embeddings = self.char_embeddings)
            self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
                                            self.question_c,
                                            word_embeddings = self.word_embeddings,
                                            char_embeddings = self.char_embeddings)

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

    def embedding_encoder(self):
        with tf.variable_scope("Embedding_Encoder_Layer"):
            self.passage_context = residual_block(self.passage_encoding,
                                                  num_blocks = 1,
                                                  num_conv_layers = 4,
                                                  kernel_size = 7,
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

    def context_to_query(self):
        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            P = tf.tile(tf.expand_dims(self.passage_context,2),[1,1,Params.max_q_len,1])
            Q = tf.tile(tf.expand_dims(self.question_context,1),[1,Params.max_p_len,1,1])
            S = tf.squeeze(trilinear([P, Q, P*Q], input_keep_prob = 1.0 - self.dropout))
            S_ = tf.nn.softmax(mask_logits(S, self.question_len))
            self.c2q_attention = tf.matmul(S_, self.question_context)
            self.c2q_attention = tf.nn.dropout(self.c2q_attention, 1.0 - self.dropout)

    def model_encoder(self):
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat([self.passage_context, self.c2q_attention, self.passage_context * self.c2q_attention], axis = -1)
            self.encoder_outputs = [conv(inputs, Params.num_units, name = "input_projection")]
            for i in range(3):
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
                    self.encoder_outputs[i + 1] = tf.nn.dropout(self.encoder_outputs[i + 1], 1.0 - self.dropout)

    def output_layer(self):
        with tf.variable_scope("Output_Layer"):
            # self.start_logits = tf.layers.dense(tf.concat([self.encoder_outputs[1], self.encoder_outputs[2]],axis = -1),1, use_bias = False, name = "start_pointer")
            # self.end_logits = tf.layers.dense(tf.concat([self.encoder_outputs[1], self.encoder_outputs[3]],axis = -1),1, use_bias = False, name = "end_pointer")
            self.start_logits = conv(tf.concat([self.encoder_outputs[1], self.encoder_outputs[2]],axis = -1),1, bias = False, name = "start_pointer")
            self.end_logits = conv(tf.concat([self.encoder_outputs[1], self.encoder_outputs[3]],axis = -1),1, bias = False, name = "end_pointer")
            logits = tf.stack([self.start_logits, self.end_logits],axis = 1)
            logits = mask_logits(tf.squeeze(logits), self.passage_len)
            self.logits = tf.nn.softmax(logits)

            self.logit_1, self.logit_2 = tf.split(self.logits, 2, axis = 1)
            self.logit_1 = tf.transpose(self.logit_1, [0, 2, 1])
            self.dp = tf.matmul(self.logit_1, self.logit_2)
            self.dp = tf.matrix_band_part(self.dp, 0, 15)
            self.output_index_1 = tf.argmax(tf.reduce_max(self.dp, axis = 2), -1)
            self.output_index_2 = tf.argmax(tf.reduce_max(self.dp, axis = 1), -1)
            self.output_index = tf.stack([self.output_index_1, self.output_index_2], axis = 1)
            # self.output_index_greedy = tf.argmax(self.logits, axis = 2)

    def loss_function(self):
        with tf.variable_scope("loss"):
            shapes = self.passage_w.shape
            self.indices_prob = tf.one_hot(self.indices, shapes[1])
            # self.mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.indices_prob))
            self.mean_loss = cross_entropy(self.logits, self.indices_prob)

            if Params.l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                self.mean_loss += l2_loss

            # apply ema
            if Params.decay is not None:
                self.var_ema = tf.train.ExponentialMovingAverage(Params.decay)
                ema = self.var_ema
                ema_op = ema.apply(tf.trainable_variables())
                with tf.control_dependencies([ema_op]):
                    self.mean_loss = tf.identity(self.mean_loss)

            # learning rate warmup scheme
            self.warmup_scheme = tf.minimum(Params.LearningRate, tf.exp(1e-6 * tf.cast(self.global_step, tf.float32)) - 1)
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
    model = Model(is_training = True)
    print("Built model")

def test():
    model = Model(is_training = False); print("Built model")
    dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    with model.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
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

def main():
    model = Model(is_training = True); print("Built model")
    dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    init = False
    devdata, dev_ind = get_dev()
    if not os.path.isfile(os.path.join(Params.logdir,"checkpoint")):
        init = True
        glove = np.memmap(Params.data_dir + "glove.np", dtype = np.float32, mode = "r")
        glove = np.reshape(glove,(Params.vocab_size,Params.emb_size))
    with model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=Params.logdir,
                        save_model_secs=0,
                        global_step = model.global_step,
                        init_op = model.init_op)
        with sv.managed_session(config = config) as sess:
            if init: sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove})
            for epoch in range(1, Params.num_epochs+1):
                if sv.should_stop(): break
                train_loss = []
                for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                    _, loss = sess.run([model.train_op, model.mean_loss],
                                        feed_dict={model.dropout: Params.dropout if Params.dropout is not None else 0.0})
                    train_loss.append(loss)
                    if step % Params.save_steps == 0:
                        gs = sess.run(model.global_step)
                        sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
                        sample = np.random.choice(dev_ind, Params.batch_size)
                        feed_dict = {data: devdata[i][sample] for i,data in enumerate(model.data)}
                        index, dev_loss = sess.run([model.output_index, model.mean_loss], feed_dict = feed_dict)
                        #index = np.argmax(logits, axis = 2)
                        F1, EM = 0.0, 0.0
                        for batch in range(Params.batch_size):
                            f1, em = f1_and_EM(index[batch], devdata[6][sample][batch], devdata[0][sample][batch], dict_)
                            F1 += f1
                            EM += em
                        F1 /= float(Params.batch_size)
                        EM /= float(Params.batch_size)
                        sess.run(model.metric_assign,{model.F1_placeholder: F1, model.EM_placeholder: EM, model.dev_loss_placeholder: dev_loss})
                        print("\nTrain_loss: {}\nDev_loss: {}\nDev_Exact_match: {}\nDev_F1_score: {}".format(np.mean(train_loss),dev_loss,EM,F1))
                        train_loss = []

if __name__ == '__main__':
    if Params.mode.lower() == "debug":
        print("Debugging...")
        debug()
    elif Params.mode.lower() == "test":
        print("Testing on dev set...")
        test()
    elif Params.mode.lower() == "train":
        print("Training...")
        main()
    else:
        print("Invalid mode.")
