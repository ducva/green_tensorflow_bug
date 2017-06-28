import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import graph_util

base_dir = os.path.realpath(os.path.dirname(__file__))

class PredictService:

    def __init__(self):
        self.forzen_session = None
        self.graph = None
        self.forzen_model_filepath = os.path.realpath(os.path.join(base_dir, './data/model/frozen_model.pb'))
        if os.path.exists(self.forzen_model_filepath):
            print("Before load graph")
            self.graph = self.load_graph()
            print("Before create session")
            self.forzen_session = tf.Session(graph=self.graph)
            print("After creating session")

    def load_graph(self):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(self.forzen_model_filepath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    def predict(self, data):
        if self.forzen_session is None:
            # We use our "load_graph" function
            print("Before Load graph")
            self.graph = self.load_graph()
            print("Before Create session")
            self.forzen_session = tf.Session(graph=self.graph)

        # We can verify that we can access the list of operations in the graph
        # for op in self.graph.get_operations():
        #     print(op.name)

        # We access the input and output nodes
        x = self.graph.get_tensor_by_name('prefix/x:0')
        model = self.graph.get_tensor_by_name('prefix/model:0')

        # We launch a Session
        print("Prepare to run Tensorflow Session")
        predictions = self.forzen_session.run(model, feed_dict={x: data})
        print("Finish run Tensorflow session")
        predictions = np.argmax(predictions, axis=1)
        return predictions.tolist()
    