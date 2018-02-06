# -*- coding: utf-8 -*-
#/usr/bin/python2

from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging

import numpy as np
import tensorflow as tf
from params import Params
from process import *

# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.
    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

def load_data(dir_):
    dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    # Target indices
    indices = load_target(dir_ + Params.target_dir)

    # Load question data
    print("Loading question data...")
    q_word_ids, q_word_len = load_word(dir_ + Params.q_word_dir)
    q_char_ids, q_char_len, _ = load_char(dir_ + Params.q_chars_dir)

    # Load passage data
    print("Loading passage data...")
    p_word_ids, p_word_len = load_word(dir_ + Params.p_word_dir)
    p_char_ids, p_char_len, _ = load_char(dir_ + Params.p_chars_dir)

    # Get max length to pad
    p_max_word = Params.max_p_len#np.max(p_word_len)
    p_max_char = Params.max_char_len#,max_value(p_char_len))
    q_max_word = Params.max_q_len#,np.max(q_word_len)
    q_max_char = Params.max_char_len#,max_value(q_char_len))

    # pad_data
    print("Preparing data...")
    p_word_ids = pad_data(p_word_ids,p_max_word)
    q_word_ids = pad_data(q_word_ids,q_max_word)
    p_char_ids = pad_char_data(p_char_ids,p_max_char,p_max_word)
    q_char_ids = pad_char_data(q_char_ids,q_max_char,q_max_word)

    # to numpy
    indices = np.reshape(np.asarray(indices,np.int32),(-1,2))

    # shapes of each data
    shapes=[(p_max_word,),(q_max_word,),
            (p_max_word,p_max_char,),(q_max_word,q_max_char,),
            (2,)]

    return ([p_word_ids, q_word_ids,
            p_char_ids, q_char_ids,
            indices], shapes)

def get_dev():
    devset, shapes = load_data(Params.dev_dir)
    indices = devset[-1]

    dev_ind = np.arange(indices.shape[0],dtype = np.int32)
    np.random.shuffle(dev_ind)
    return devset, dev_ind

def get_batch(is_training = True):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load dataset
        input_list, shapes = load_data(Params.train_dir if is_training else Params.dev_dir)
        indices = input_list[-1]

        train_ind = np.arange(indices.shape[0],dtype = np.int32)
        np.random.shuffle(train_ind)

        size = Params.data_size
        if Params.data_size > indices.shape[0] or Params.data_size == -1:
            size = indices.shape[0]
        ind_list = tf.convert_to_tensor(train_ind[:size])

        # Create Queues
        ind_list = tf.train.slice_input_producer([ind_list], shuffle=True)

        @producer_func
        def get_data(ind):
            '''From `_inputs`, which has been fetched from slice queues,
               then enqueue them again.
            '''
            return [np.reshape(input_[ind], shapes[i]) for i,input_ in enumerate(input_list)]

        data = get_data(inputs=ind_list,
                        dtypes=[np.int32]*5,
                        capacity=Params.batch_size*8,
                        num_threads=2)

        # create batch queues
        batch = tf.train.batch(data,
                                shapes=shapes,
                                num_threads=2,
                                batch_size=Params.batch_size,
                                capacity=Params.batch_size*8,
                                dynamic_pad=True)

    return batch, size // Params.batch_size
