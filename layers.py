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

initializer = tf.contrib.layers.xavier_initializer
initializer_conv2d = tf.contrib.layers.xavier_initializer_conv2d

def encoding(word, char, word_embeddings, char_embeddings, scope = "embedding"):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding

def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, num_filters, scope = "res_block", is_training = True, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        inputs = tf.layers.dense(inputs, num_filters, name = "input_projection", reuse = reuse)
        outputs = tf.contrib.layers.layer_norm(inputs, scope = "layer_norm", reuse = reuse)
        for i in range(num_blocks):
            outputs = encoder_block(outputs, num_conv_layers, kernel_size, num_filters, scope = "encoder_block_%d"%i,reuse = reuse)
        outputs += inputs
        return outputs

def encoder_block(inputs, num_conv_layers, kernel_size, num_filters, scope = "encoder_block", is_training = True, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        inputs = add_timing_signal_1d(inputs)
        outputs = depthwise_separable_convolution(inputs, num_layers = num_conv_layers, kernel_size = kernel_size, num_filters = num_filters, reuse = reuse)
        outputs = self_attention(outputs, num_filters, num_heads = 8, reuse = reuse, is_training = is_training)
        return tf.layers.dense(outputs, num_filters, activation = tf.nn.relu, name = "output_projection", reuse = reuse)

def self_attention(queries, units, num_heads, scope = "Multi_Head_Attention", reuse = None, is_training = True):
    with tf.variable_scope(scope, reuse = reuse):
        combined = tf.layers.dense(queries, 3 * units, activation = tf.nn.relu, reuse = reuse)
        Q, K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(combined,3,axis = 2)]
        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5  # [batch, length_q, head, depth_k]
        Q = tf.transpose(Q,[0, 2, 1, 3]) # [batch, heads, length_q, depth_k]
        K = tf.transpose(K,[0, 2, 1, 3])
        V = tf.transpose(V,[0, 2, 1, 3])
        x = dot_product_attention(Q,K,V,bias = True, dropout_rate = Params.dropout, is_training = is_training, scope = "dot_product_attention", reuse = reuse) # [batch, heads, length, depth]
        x = tf.transpose(x, [0, 2, 1, 3])
        return combine_last_two_dimensions(x)

def mask_logits(inputs, sequence_length, mask_value = -1e30):
    return inputs
    # shapes = inputs.shape.as_list()
    # input_mask = tf.expand_dims(tf.sequence_mask(
    #     sequence_length, maxlen=shapes[-1]),1)
    # input_mask = tf.tile(input_mask,[1,shapes[1],1])
    # mask_values = mask_value * tf.ones_like(inputs)
    # masked_outputs = tf.where(input_mask, inputs, mask_values)
    # return tf.reshape(masked_outputs,(shapes[0],shapes[1],shapes[2]))

def depthwise_separable_convolution(inputs, num_layers, kernel_size, num_filters, scope = "depthwise_separable_convolution", reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        outputs = tf.expand_dims(inputs, axis = 2)
        for i in range(1,num_layers):
            shapes = outputs.shape.as_list()
            depthwise_filter = tf.get_variable("depthwise_filter_%d"%i, (kernel_size, 1, shapes[-1], 1), dtype = tf.float32, initializer = initializer_conv2d())
            pointwise_filter = tf.get_variable("pointwise_filter_%d"%i, (1,1,shapes[-1],num_filters), dtype = tf.float32, initializer = initializer_conv2d())
            outputs = tf.nn.separable_conv2d(outputs,
                                depthwise_filter,
                                pointwise_filter,
                                strides = (1,1,1,1),
                                padding = "SAME")
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
    return ret

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          is_training = True,
                          scope=None,
                          reuse = None):
    """dot-product attention.
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    is_training: a bool of training
    scope: an optional string
    Returns:
    A Tensor.
    """
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        # print(logits)
        # if bias:
        #     b = tf.get_variable("bias", logits.shape[-1], initializer = initializer())
        #     logits += b
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        if is_training and Params.dropout:
            weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
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

def trilinear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    flat_args = [flatten(arg, 1) for arg in args]
    if input_keep_prob < 1.0:
        if is_train:
            flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     for arg in flat_args]
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
            bias_initializer=None,
            scope = None,
            kernel_initializer=None):
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
  with tf.variable_scope(scope) as outer_scope:
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
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
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
