import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, graph, a=5.0, b=6.0):
        self.graph   = self.build_graph(a, b)
        self.session = tf.Session()#graph=self.graph)

    def build_graph(self, a, b):
        # with self.graph.as_default():
        self.a = tf.constant(a)
        self.b = tf.constant(b)
        self.c = a * b
        print(tf.get_default_graph())
        print(self.a, self.b, self.c)

    def run(self):
        out = self.session.run(self.a)
        print(out)
        return out


graphs = {'g_1': tf.Graph(),
          'g_2': tf.Graph()
          }

model1 = Model(graphs['g_1'])
model2 = Model(graphs['g_2'])

model1.build_graph(5, 6)
model2.build_graph(12, 6)

model1.run()
model2.run()

