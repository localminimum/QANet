#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import bottle
from bottle import route, run
import threading
import json
import numpy as np

from prepro import convert_to_features, word_tokenize
from time import sleep

'''
This file is taken and modified from R-Net by Minsangkim142
https://github.com/minsangkim142/R-net
'''

app = bottle.Bottle()
query = []
response = ""

@app.get("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html

@app.post('/answer')
def answer():
    passage = bottle.request.json['passage']
    question = bottle.request.json['question']
    print("received question: {}".format(question))
    # if not passage or not question:
    #     exit()
    global query, response
    query = (passage, question)
    while not response:
        sleep(0.1)
    print("received response: {}".format(response))
    response_ = {"answer": response}
    response = []
    return response_

class Demo(object):
    def __init__(self, model, config):
        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.demo_backend, args = [model, config, run_event]).start()
        app.run(port=8080, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()

    def demo_backend(self, model, config, run_event):
        global query, response

        with open(config.word_dictionary, "r") as fh:
            word_dictionary = json.load(fh)
        with open(config.char_dictionary, "r") as fh:
            char_dictionary = json.load(fh)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with model.graph.as_default():

            with tf.Session(config=sess_config) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
                if config.decay < 1.0:
                    sess.run(model.assign_vars)
                while run_event.is_set():
                    sleep(0.1)
                    if query:
                        context = word_tokenize(query[0].replace("''", '" ').replace("``", '" '))
                        c,ch,q,qh = convert_to_features(config, query, word_dictionary, char_dictionary)
                        fd = {'context:0': [c],
                              'question:0': [q],
                              'context_char:0': [ch],
                              'question_char:0': [qh]}
                        yp1,yp2 = sess.run([model.yp1, model.yp2], feed_dict = fd)
                        if yp1 == yp2:
                            yp2[0] += 1
                        response = " ".join(context[yp1[0]:yp2[0]])
                        query = []
