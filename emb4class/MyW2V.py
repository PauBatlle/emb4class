import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sklearn.preprocessing as pre
from collections import Counter
from tqdm import tqdm as t



class Word2Vec():
    def __init__ (self, pairs_file, vocab_path, epochs, batch_size, embedding_size, k, learning_rate=1):
        self.pairs_file = pairs_file
        self.pairs = self.read_file()
        self.vocabulary_path = vocab_path
        self.vocabulary = self.obtain_vocabulary()
        self.voc_size = np.max(self.pairs.reshape(-1))+1#len(vocabulary)
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.k = k
        self.learning_rate = learning_rate
        self.create_graph()
        self.train()

    def obtain_vocabulary(self):
        """ Read vocabulary from a file """
        with open(self.vocabulary_path, "r") as rfile:
            #llegeix
            pass


        assert len(vocabulary) == np.max(self.pairs.reshape(-1))+1
    def read_file(self):
        """Read the number pairs from a .txt or .npy (from numpy.save)"""
        _ , file_extension = os.path.splitext(self.pairs_file)
        if file_extension == ".npy":
            return np.load(self.pairs_file).astype(int)
        else:
            try:
                #Try numpy function to read txts
                return np.loadtxt(path).astype(int)
            except:
                pairs = []
                with open(self.pairs_file, "r") as file_to_read:
                    for pair in file_to_read:
                        #int(float()) in case it is written in scientific notation,
                        #such as the one that numpy.savetxt uses
                        pairs.append([int(float(i)) for i in pair.split()])
                return np.array(pairs).astype(int)
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

        def generate_batch(size, pairs):
            """Generate a minibatch of a given size """
            assert size < len(pairs)
            r = np.random.choice(range(len(pairs)), size, replace=False)
            #The Word2Vec Graph needs each label as a vector of one element
            return [pairs[i][0] for i in r], [[pairs[i][1]] for i in r]

        with self.sess as session:
            tf.global_variables_initializer().run()
            results = []
            for epoch in range(self.epochs):
                batch_inputs, batch_labels = generate_batch(self.batch_size, self.pairs)
                _, loss_val = self.sess.run([self.train_op, self.loss], feed_dict={self.train_inputs: batch_inputs, self.train_labels: batch_labels})
                print("Loss at ",epoch, loss_val) # Report the loss
                results += [loss_val]
                plt.plot(results)
                plt.savefig("newloss.png")
                plt.close()
            # Final embeddings are ready to use
            self.trained_embeddings = self.embeddings.eval()
    def save(self):
        """ Save the most important parameters of the model in a .txt file """
        return

def load(path):
    """ Load a model that was previously saved using the save function """
    with open(path, "r") as file:
        return


p = Word2Vec("pairs.npy", 1000, 200, 100, 10)
