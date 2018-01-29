# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
import numpy as np
import math

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops

from params import Params
from functools import reduce
from operator import mul
# from common_layers import *

'''
Some of the functions are borrowed from Tensor2Tensor Library https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
and BiDAF repository https://github.com/allenai/bi-att-flow.
'''

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False, seed=None, dtype=tf.float32)
initializer_relu = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)

def encoding(word, char, word_embeddings, char_embeddings, scope = "embedding"):
	word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
	char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
	return word_encoding, char_encoding

def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, num_filters, input_projection = False, seq_len = None, scope = "res_block", is_training = True, reuse = None, bias = Params.bias):
	with tf.variable_scope(scope, reuse = reuse):
		if input_projection:
			inputs = conv(inputs, num_filters, name = "input_projection", reuse = reuse)
		outputs = inputs
		for i in range(num_blocks):
			outputs = add_timing_signal_1d(outputs)
			outputs = conv_block(outputs, num_conv_layers, kernel_size, num_filters, seq_len = seq_len, scope = "encoder_block_%d"%i,reuse = reuse, bias = bias)
			outputs = self_attention_block(outputs, num_filters, seq_len, scope = "self_attention_layers%d"%i, reuse = reuse, is_training = is_training, bias = bias)
			if Params.dropout is not None and is_training:
				if (i + 1) % 2 == 0:
					outputs = tf.nn.dropout(outputs, 1.0 - Params.dropout)
		return outputs

def conv_block(inputs, num_conv_layers, kernel_size, num_filters, seq_len = None, scope = "conv_block", is_training = True, reuse = None, bias = Params.bias):
	with tf.variable_scope(scope, reuse = reuse):
		outputs = inputs
		for i in range(num_conv_layers):
			residual = outputs
			outputs = tf.contrib.layers.layer_norm(residual, scope = "layer_norm_%d"%i, reuse = reuse)
			outputs = depthwise_separable_convolution(outputs, kernel_size = kernel_size, num_filters = num_filters, scope = "depthwise_conv_layers_%d"%i, is_training = is_training, reuse = reuse) + residual
		return outputs

def self_attention_block(inputs, num_filters, seq_len, scope = "self_attention_ffn", reuse = None, is_training = True, bias = Params.bias):
	with tf.variable_scope(scope, reuse = reuse):
		outputs = tf.contrib.layers.layer_norm(inputs, scope = "layer_norm_1", reuse = reuse)
		outputs = multihead_attention(outputs, num_filters, num_heads = Params.num_heads, seq_len = seq_len, reuse = reuse, is_training = is_training, bias = bias)
		residual = outputs + inputs
		outputs = tf.contrib.layers.layer_norm(residual, scope = "layer_norm_2", reuse = reuse)
		return conv(outputs, num_filters, bias, tf.nn.relu, name = "FFN", reuse = reuse) + residual

def multihead_attention(queries, units, num_heads, seq_len = None, scope = "Multi_Head_Attention", reuse = None, is_training = True, bias = Params.bias):
	with tf.variable_scope(scope, reuse = reuse):
		combined = conv(queries, 3 * units, name = "projection", reuse = reuse)
		Q, K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(combined,3,axis = 2)]
		key_depth_per_head = units // num_heads
		Q *= key_depth_per_head**-0.5
		x = dot_product_attention(Q,K,V,bias = bias, seq_len = seq_len, is_training = is_training, scope = "dot_product_attention", reuse = reuse)
		# Apply branched attention from https://arxiv.org/pdf/1711.02132v1.pdf
		if Params.attention == "branched":
			shapes = x.shape.as_list()
			kappa = tf.reshape(tf.nn.softmax(tf.get_variable("kappa", num_heads, dtype = tf.float32, initializer = tf.random_uniform_initializer())), (1, num_heads, 1, 1))
			alpha = tf.reshape(tf.nn.softmax(tf.get_variable("alpha", num_heads, dtype = tf.float32, initializer = tf.random_uniform_initializer())), (1, num_heads, 1, 1))
			x = conv(x, units, name = "output_projection", reuse = reuse) * kappa
			x = conv(x, units, bias = True, activation = tf.nn.relu, name ="Feed_forward_network", reuse = reuse) * alpha
			return tf.reduce_sum(x, axis = 1) + queries
		else:
			return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))

def conv(inputs, output_size, bias = None, activation = None, name = "conv", reuse = None):
	with tf.variable_scope(name, reuse = reuse):
		shapes = inputs.shape.as_list()
		if len(shapes) > 4:
			raise NotImplementedError
		elif len(shapes) == 4:
			filter_shape = [1,1,shapes[-1],output_size]
			bias_shape = [1,1,1,output_size]
			strides = [1,1,1,1]
		else:
			filter_shape = [1,shapes[-1],output_size]
			bias_shape = [1,1,output_size]
			strides = 1
		conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
		kernel_ = tf.get_variable("kernel_", filter_shape, dtype = tf.float32, initializer = initializer_relu if activation is not None else initializer)
		outputs = conv_func(inputs, kernel_, strides, "VALID")
		if bias:
			outputs += tf.get_variable("bias_", bias_shape, initializer = initializer_relu if activation is not None else initializer)
		if activation is not None:
			return activation(outputs)
		else:
			return outputs

def mask_logits(inputs, sequence_length, mask_value = -1e7):
	shapes = inputs.shape.as_list()
	mask = tf.reshape(tf.sequence_mask(sequence_length, maxlen=shapes[-1], dtype = tf.float32),[-1,1,1,shapes[-1]] if len(shapes) == 4 else [-1,1,shapes[-1]])
	mask_values = mask_value * tf.to_float(tf.not_equal(mask, tf.ones_like(mask)))
	return inputs + mask_values
	# return inputs

def cross_entropy(output, target):
	cross_entropy = target * tf.log(output + 1e-7)
	cross_entropy = -tf.reduce_sum(cross_entropy, [1,2])
	return tf.reduce_mean(cross_entropy)

def depthwise_separable_convolution(inputs, kernel_size, num_filters, scope = "depthwise_separable_convolution", is_training = True, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		outputs = tf.expand_dims(inputs, axis = 2)
		# for i in range(num_layers):
		shapes = outputs.shape.as_list()
		depthwise_filter = tf.get_variable("depthwise_filter", (kernel_size, 1, shapes[-1], 1), dtype = tf.float32, initializer = initializer_relu)
		pointwise_filter = tf.get_variable("pointwise_filter", (1,1,shapes[-1],num_filters), dtype = tf.float32, initializer = initializer_relu)
		outputs = tf.nn.separable_conv2d(outputs,
							depthwise_filter,
							pointwise_filter,
							strides = (1,1,1,1),
							padding = "SAME")
		outputs = tf.nn.relu(outputs)
		return tf.squeeze(outputs)

def split_last_dimension(x, n):
	"""Reshape x so that the last dimension becomes two dimensions.
	The first of these two dimensions is n.
	Args:
	x: a Tensor with shape [..., m]
	n: an integer.
	Returns:
	a Tensor with shape [..., n, m/n]
	"""
	old_shape = x.get_shape().dims
	last = old_shape[-1]
	new_shape = old_shape[:-1] + [n] + [last // n if last else None]
	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
	ret.set_shape(new_shape)
	return tf.transpose(ret,[0,2,1,3])

def dot_product_attention(q,
						  k,
						  v,
						  bias,
						  seq_len = None,
						  is_training = True,
						  scope=None,
						  reuse = None):
	"""dot-product attention.
	Args:
	q: a Tensor with shape [batch, heads, length_q, depth_k]
	k: a Tensor with shape [batch, heads, length_kv, depth_k]
	v: a Tensor with shape [batch, heads, length_kv, depth_v]
	bias: bias Tensor (see attention_bias())
	is_training: a bool of training
	scope: an optional string
	Returns:
	A Tensor.
	"""
	with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
		# [batch, num_heads, query_length, memory_length]
		logits = tf.matmul(q, k, transpose_b=True)
		if bias:
			b = tf.get_variable("bias", logits.shape[-1], initializer = initializer)
			logits += b
		if seq_len is not None:
			logits = mask_logits(logits, seq_len)
		weights = tf.nn.softmax(logits, name="attention_weights")
		# dropping out the attention links for each of the heads
		if is_training and Params.dropout is not None:
			weights = tf.nn.dropout(weights, 1.0 - Params.dropout)
		return tf.matmul(weights, v)

def combine_last_two_dimensions(x):
	"""Reshape x so that the last two dimension become one.
	Args:
	x: a Tensor with shape [..., a, b]
	Returns:
	a Tensor with shape [..., ab]
	"""
	old_shape = x.get_shape().dims
	a, b = old_shape[-2:]
	new_shape = old_shape[:-2] + [a * b if a and b else None]
	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
	ret.set_shape(new_shape)
	return ret

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
	"""Adds a bunch of sinusoids of different frequencies to a Tensor.
	Each channel of the input Tensor is incremented by a sinusoid of a different
	frequency and phase.
	This allows attention to learn to use absolute and relative positions.
	Timing signals should be added to some precursors of both the query and the
	memory inputs to attention.
	The use of relative position is possible because sin(x+y) and cos(x+y) can be
	experessed in terms of y, sin(x) and cos(x).
	In particular, we use a geometric sequence of timescales starting with
	min_timescale and ending with max_timescale.  The number of different
	timescales is equal to channels / 2. For each timescale, we
	generate the two sinusoidal signals sin(timestep/timescale) and
	cos(timestep/timescale).  All of these sinusoids are concatenated in
	the channels dimension.
	Args:
	x: a Tensor with shape [batch, length, channels]
	min_timescale: a float
	max_timescale: a float
	Returns:
	a Tensor the same shape as x.
	"""
	length = tf.shape(x)[1]
	channels = tf.shape(x)[2]
	signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
	return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
	"""Gets a bunch of sinusoids of different frequencies.
	Each channel of the input Tensor is incremented by a sinusoid of a different
	frequency and phase.
	This allows attention to learn to use absolute and relative positions.
	Timing signals should be added to some precursors of both the query and the
	memory inputs to attention.
	The use of relative position is possible because sin(x+y) and cos(x+y) can be
	experessed in terms of y, sin(x) and cos(x).
	In particular, we use a geometric sequence of timescales starting with
	min_timescale and ending with max_timescale.  The number of different
	timescales is equal to channels / 2. For each timescale, we
	generate the two sinusoidal signals sin(timestep/timescale) and
	cos(timestep/timescale).  All of these sinusoids are concatenated in
	the channels dimension.
	Args:
	length: scalar, length of timing signal sequence.
	channels: scalar, size of timing embeddings to create. The number of
		different timescales is equal to channels / 2.
	min_timescale: a float
	max_timescale: a float
	Returns:
	a Tensor of timing signals [1, length, channels]
	"""
	position = tf.to_float(tf.range(length))
	num_timescales = channels // 2
	log_timescale_increment = (
		math.log(float(max_timescale) / float(min_timescale)) /
			(tf.to_float(num_timescales) - 1))
	inv_timescales = min_timescale * tf.exp(
		tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
	scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
	signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
	signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
	signal = tf.reshape(signal, [1, length, channels])
	return signal

def trilinear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob= 1.0,
		   is_training=None):
	flat_args = [flatten(arg, 1) for arg in args]
	if input_keep_prob < 1.0 and is_training:
		flat_args = [tf.nn.dropout(arg, input_keep_prob,noise_shape = [1,output_size]) for arg in flat_args]
	flat_out = _linear(flat_args, output_size, bias, scope=scope)
	out = reconstruct(flat_out, args[0], 1)
	if squeeze:
		out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
	return out

def flatten(tensor, keep):
	fixed_shape = tensor.get_shape().as_list()
	start = len(fixed_shape) - keep
	left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
	out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
	flat = tf.reshape(tensor, out_shape)
	return flat

def reconstruct(tensor, ref, keep):
	ref_shape = ref.get_shape().as_list()
	tensor_shape = tensor.get_shape().as_list()
	ref_stop = len(ref_shape) - keep
	tensor_start = len(tensor_shape) - keep
	pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
	keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
	# pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
	# keep_shape = tensor.get_shape().as_list()[-keep:]
	target_shape = pre_shape + keep_shape
	out = tf.reshape(tensor, target_shape)
	return out

def _linear(args,
			output_size,
			bias,
			bias_initializer=initializer,
			scope = None,
			kernel_initializer=initializer,
			reuse = None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
	args: a 2D Tensor or a list of 2D, batch x n, Tensors.
	output_size: int, second dimension of W[i].
	bias: boolean, whether to add a bias term or not.
	bias_initializer: starting value to initialize the bias
	  (default is all zeros).
	kernel_initializer: starting value to initialize the weight.
  Returns:
	A 2D Tensor with shape [batch x output_size] equal to
	sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
	ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
	raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
	args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
	if shape.ndims != 2:
	  raise ValueError("linear is expecting 2D arguments: %s" % shapes)
	if shape[1].value is None:
	  raise ValueError("linear expects shape[1] to be provided for shape %s, "
					   "but saw %s" % (shape, shape[1]))
	else:
	  total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope, reuse = reuse) as outer_scope:
	weights = tf.get_variable(
		"linear_kernel", [total_arg_size, output_size],
		dtype=dtype,
		initializer=kernel_initializer)
	if len(args) == 1:
	  res = math_ops.matmul(args[0], weights)
	else:
	  res = math_ops.matmul(array_ops.concat(args, 1), weights)
	if not bias:
	  return res
	with tf.variable_scope(outer_scope) as inner_scope:
	  inner_scope.set_partitioner(None)
	  biases = tf.get_variable(
		  "linear_bias", [output_size],
		  dtype=dtype,
		  initializer=bias_initializer)
	return nn_ops.bias_add(res, biases)

def total_params():
	total_parameters = 0
	for variable in tf.trainable_variables():
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	print("Total number of trainable parameters: {}".format(total_parameters))
