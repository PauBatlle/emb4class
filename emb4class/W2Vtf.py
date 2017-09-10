import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sklearn.preprocessing as pre
from collections import Counter
from tqdm import tqdm as t
import re, string
from nltk import word_tokenize
from collections import Counter
import random

class Word2Vec():
    def __init__ (self, pairs, word2vec_options, directory):
        self.pairs = pairs
        self.directory = directory
        #Length of vocabulary
        self.voc_size = np.max(self.pairs.reshape(-1))+1
        self.opt = word2vec_options
        self.learning_rate = self.opt.learning_rate
        self.total_batches = self.opt.total_batches
        self.display_step = self.opt.display_step
        self.embedding_size = self.opt.embedding_size
        self.batch_size = self.opt.batch_size
        self.k = self.opt.k
        self.create_graph()
        self.train()

    def create_graph(self):
        """ Create the graph for training the Word2Vec """
        self.sess = tf.Session()
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        # It is needed to shape [batch_size, 1] for nn.nce_loss
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        #Layer that goes from inital vectors to the embeddings
        self.embeddings = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        #Lookup table
        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        #Weights and biases for the loss function
        self.nce_weights = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        self.nce_biases = tf.Variable(tf.zeros([self.voc_size]))
        #Loss function. It automatically takes care of the negative sampling
        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.train_labels,
                                                  self.embed, self.k, self.voc_size))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        """ Train the model """

        def generate_average(vec, param = 10):
            if len(vec) < 10:
                return np.mean(vec[-param:])
            else:
                return np.mean(vec[-10:])

        def generate_batch(size, pairs):
            """Generate a minibatch of a given size """
            assert size < len(pairs)
            r = random.sample(range(len(pairs)),size)
            #The Word2Vec Graph needs each label as a vector of one element
            aux = pairs[r]
            return [i[0] for i in aux], [[i[1]] for i in aux]

        with self.sess as session:
            tf.global_variables_initializer().run()
            results = []
            averages = []
            averages2 = []
            for batch in t(range(self.total_batches)):
                batch_inputs, batch_labels = generate_batch(self.batch_size, self.pairs)
                _, loss_val = session.run([self.train_op, self.loss], feed_dict={self.train_inputs: batch_inputs, self.train_labels: batch_labels})
                if batch % self.display_step == 0:
                    print("Loss at ", batch ,":", loss_val) # Report the loss
                results += [loss_val]
                averages += [np.mean(results[-25:])]
                averages2 += [np.mean(results[-100:])]
                plt.plot(results, label = 'loss at each batch')
                plt.plot(averages, label = '25 batch average')
                plt.plot(averages2, label = '100 batch average')
                plt.legend()
                plt.savefig(self.directory+"/newloss.png")
                plt.close()
            # Final embeddings are ready to use
            self.trained_embeddings = self.embeddings.eval()
