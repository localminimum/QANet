#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import bottle
from bottle import route, run
import threading

from params import Params
from process import *
from time import sleep

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
    def __init__(self, model):
        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.demo_backend, args = [model, run_event]).start()
        app.run(port=8080, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print "Closing server..."
            run_event.clear()

    def demo_backend(self, model, run_event):
        global query, response
        dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))

        with model.graph.as_default():
            sv = tf.train.Supervisor()
            with sv.managed_session() as sess:
                sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
                while run_event.is_set():
                    sleep(0.1)
                    if query:
                        data, shapes = dict_.realtime_process(query)
                        fd = {m:d for i,(m,d) in enumerate(zip(model.data, data))}
                        ids, confidence = sess.run([model.output_index, model.dp], feed_dict = fd)
                        ids = ids[0]
                        confidence = confidence[0]
                        if ids[0] == ids[1]:
                            ids[1] += 1
                        passage_t = tokenize(query[0])
                        response = " ".join(passage_t[ids[0]:ids[1]])
                        query = []
