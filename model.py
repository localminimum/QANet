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
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.data, self.num_batch = get_batch(is_training = is_training)
			(self.passage_w,
			self.question_w,
			self.passage_c,
			self.question_c,
			self.passage_w_len_,
			self.question_w_len_,
			self.passage_c_len,
			self.question_c_len,
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
			self.unknown_word_embeddings = tf.get_variable("unknown_word_embeddings", (1, Params.emb_size),dtype = tf.float32, initializer = initializer())
			self.unknown_char_embeddings = tf.get_variable("unknown_char_embeddings", (1, Params.emb_size),dtype = tf.float32, initializer = initializer())
			self.char_embeddings = tf.get_variable("char_embeddings", (Params.char_vocab_size, Params.emb_size), dtype = tf.float32, initializer = initializer())
			self.char_embeddings = tf.concat((self.unknown_char_embeddings, self.char_embeddings),axis = 0)
			with tf.device('/cpu:0'):
				self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
				self.word_embeddings_placeholder = tf.placeholder(tf.float32,[Params.vocab_size, Params.emb_size],"word_embeddings_placeholder")
				self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)
				self.word_embeddings = tf.concat((self.unknown_word_embeddings, self.word_embeddings),axis = 0)

			# Embed the question and passage information for word and character tokens
			self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
											self.passage_c,
											word_embeddings = self.word_embeddings,
											char_embeddings = self.char_embeddings,
											scope = "passage_embeddings")
			self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
											self.question_c,
											word_embeddings = self.word_embeddings,
											char_embeddings = self.char_embeddings,
											scope = "question_embeddings")

			self.passage_char_encoded = tf.reduce_max(self.passage_char_encoded, axis = 2)
			self.question_char_encoded = tf.reduce_max(self.question_char_encoded, axis = 2)
			self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded), axis = -1)
			self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded), axis = -1)

	def embedding_encoder(self):
		with tf.variable_scope("Embedding_Encoder_Layer"):
			self.passage_context = residual_block(self.passage_encoding, num_blocks = 1, num_conv_layers = 4, kernel_size = 7 , num_filters = Params.num_units, scope = "Encoder_Residual_Block", is_training = self.is_training, reuse = False)
			self.question_context = residual_block(self.question_encoding, num_blocks = 1, num_conv_layers = 4, kernel_size = 7 , num_filters = Params.num_units, scope = "Encoder_Residual_Block", is_training = self.is_training, reuse = True)

	def context_to_query(self):
		with tf.variable_scope("Context_to_Query_Attention_Layer"):
			P = tf.tile(tf.expand_dims(self.passage_context,2),[1,1,Params.max_q_len,1])
			Q = tf.tile(tf.expand_dims(self.question_context,1),[1,Params.max_p_len,1,1])
			S = tf.squeeze(trilinear([P, Q, P*Q], 1, bias = Params.bias, scope = "trilinear"))
			S_ = tf.nn.softmax(mask_logits(S, self.question_len))
			self.c2q_attention = tf.matmul(S_, self.question_context)

	def model_encoder(self):
		with tf.variable_scope("Model_Encoder_Layer"):
			inputs = tf.concat([self.passage_context, self.c2q_attention, self.passage_context * self.c2q_attention], axis = -1)
			self.encoder_outputs = [tf.layers.dense(inputs, Params.num_units, name = "input_projection")]
			for i in range(3):
				self.encoder_outputs.append(residual_block(self.encoder_outputs[i], num_blocks = 7, num_conv_layers = 2, kernel_size = 5, num_filters = Params.num_units, scope = "Model_Encoder", reuse = True if i > 0 else None))

	def output_layer(self):
		with tf.variable_scope("Output_Layer"):
			start_prob = tf.layers.dense(tf.concat([self.encoder_outputs[1], self.encoder_outputs[2]],axis = -1),1, name = "start_index_projection")
			end_prob = tf.layers.dense(tf.concat([self.encoder_outputs[1], self.encoder_outputs[3]],axis = -1),1, name = "end_index_projection")
			logits = tf.stack([start_prob, end_prob],axis = 1)
			self.logits = mask_logits(tf.squeeze(logits), self.passage_len)

	def loss_function(self):
		with tf.variable_scope("loss"):
			shapes = self.passage_w.shape
			self.indices_prob = tf.one_hot(self.indices, shapes[1])
			self.mean_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.indices_prob, logits = self.logits),axis = -1))
			self.optimizer = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])

			if Params.clip:
				# gradient clipping by norm
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
		self.dev_loss = tf.Variable(tf.constant(5.0, shape=(), dtype = tf.float32),trainable=False, name="dev_loss")
		self.dev_loss_placeholder = tf.placeholder(tf.float32, shape = (), name = "dev_loss")
		self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder),tf.assign(self.dev_loss, self.dev_loss_placeholder))
		tf.summary.scalar('loss_training', self.mean_loss)
		tf.summary.scalar('loss_dev', self.dev_loss)
		tf.summary.scalar("F1_Score",self.F1)
		tf.summary.scalar("Exact_Match",self.EM)
		tf.summary.scalar('learning_rate', Params.opt_arg[Params.optimizer]['learning_rate'])
		self.merged = tf.summary.merge_all()

def debug():
	model = Model(is_training = True)
	print("Built model")

# def test():
# 	model = Model(is_training = False); print("Built model")
# 	dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
# 	with model.graph.as_default():
# 		sv = tf.train.Supervisor()
# 		with sv.managed_session() as sess:
# 			sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
# 			EM, F1 = 0.0, 0.0
# 			for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
# 				index, ground_truth, passage = sess.run([model.output_index, model.indices, model.passage_w])
# 				for batch in range(Params.batch_size):
# 					f1, em = f1_and_EM(index[batch], ground_truth[batch], passage[batch], dict_)
# 					F1 += f1
# 					EM += em
# 			F1 /= float(model.num_batch * Params.batch_size)
# 			EM /= float(model.num_batch * Params.batch_size)
# 			print("Exact_match: {}\nF1_score: {}".format(EM,F1))

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
				for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
					sess.run(model.train_op)
					if step % Params.save_steps == 0:
						gs = sess.run(model.global_step)
						sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
						sample = np.random.choice(dev_ind, Params.batch_size)
						feed_dict = {data: devdata[i][sample] for i,data in enumerate(model.data)}
						logits, dev_loss = sess.run([model.logits, model.mean_loss], feed_dict = feed_dict)
						index = np.argmax(logits, axis = 2)
						F1, EM = 0.0, 0.0
						for batch in range(Params.batch_size):
							f1, em = f1_and_EM(index[batch], devdata[8][sample][batch], devdata[0][sample][batch], dict_)
							F1 += f1
							EM += em
						F1 /= float(Params.batch_size)
						EM /= float(Params.batch_size)
						sess.run(model.metric_assign,{model.F1_placeholder: F1, model.EM_placeholder: EM, model.dev_loss_placeholder: dev_loss})
						print("\nDev_loss: {}\nDev_Exact_match: {}\nDev_F1_score: {}".format(dev_loss,EM,F1))

# def get_best_index(logits):
# 	p1,p2 = [np.squeeze(logit) for logit in np.split(logits,2,axis=1)]
# 	i_1 = np.argmax(p1, axis = -1)
# 	i_2 = np.argmax(p2, axis = -1)
# 	indices = np.where(i_1 > i_2)[0].tolist()
# 	for i in indices:
# 		p1_ = np.reshape(p1[i],(-1,1))
# 		p2_ = np.reshape(p2[i],(1,-1))
# 		mat_ = np.matmul(p1_,p2_)

if __name__ == '__main__':
	if Params.mode.lower() == "debug":
		print("Debugging...")
		debug()
	# elif Params.mode.lower() == "test":
	# 	print("Testing on dev set...")
	# 	test()
	elif Params.mode.lower() == "train":
		print("Training...")
		main()
	else:
		print("Invalid mode.")
