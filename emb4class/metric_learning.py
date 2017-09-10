import numpy as np
from tqdm import tqdm as t
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import embed
#We assume X already normalized

class Metric_Learning():
    """Class to perform Metric_Learning"""
    def __init__(self, options, embedded_sentences, labels, directory):
        self.opt = options
        self.dir = directory
        self.X = embedded_sentences
        self.Y = labels
        self.create_graph()
        self.train()

    def create_graph(self):
        self.nsamples, self.dim = self.X.shape
        self.qry = tf.placeholder(tf.float32, [1, self.dim])
        self.pos = tf.placeholder(tf.float32, [1, self.dim])
        self.neg = tf.placeholder(tf.float32, [1, self.dim])

        # The embedding parameters
        # Projection matrix
        self.W = tf.Variable(tf.random_normal([self.dim, self.opt.embedding_size]))
        # We also add a bias term
        self.b = tf.Variable(tf.zeros([self.opt.embedding_size]))

        # Function to embed the inputs
        self.eqry = tf.matmul(self.qry, self.W) + self.b
        self.epos = tf.matmul(self.pos, self.W) + self.b
        self.eneg = tf.matmul(self.neg, self.W) + self.b

        # Define triplet loss in the space of the embeddings
        self.cost = tf.maximum(0.0, 1 + tf.matmul(self.eqry, tf.transpose(tf.subtract(self.eneg, self.epos))))
        # Gradient Descent Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.opt.learning_rate).minimize(self.cost)
        return

    def train(self):
        # Launch the graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Training cycle
            avg_costs = []
            for epoch in t(range(self.opt.training_epochs)):
                avg_cost = 0
                # Loop over all samples
                for i in t(np.arange(0, self.nsamples)):
                    # Each "input sample" consists of a "query", a positive and a negative example
                    # This is constructed on the fly
                    # Ideally, this should be precomputed
                    # Query: take the next training example
                    x_qry = self.X[i,None]
                    # Positive: sample a training example from the same class as the query
                    idx  = np.where(self.Y == self.Y[i])[0]
                    r = np.random.randint(len(idx))
                    ipos = idx[r]
                    x_pos = self.X[ipos,None]
                    # Negative: sample a training example with a different class label
                    idx  = np.where(np.logical_not(self.Y == self.Y[i]))[0]
                    r = np.random.randint(len(idx))
                    ineg = idx[r]
                    x_neg = self.X[ineg,None]
                    # Perform gradient step
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.qry: x_qry, self.pos: x_pos, self.neg: x_neg})
                    # Update loss of this epoch
                    avg_cost += c/self.nsamples

                # Display logs per epoch step
                if epoch % self.opt.display_step:
                    print("Cost at epoch",epoch,"=",avg_cost)
                avg_costs.append(avg_cost.reshape(-1)[0])
                plt.plot(avg_costs)
                plt.savefig(self.dir+"loss.png")
                plt.close()
            self.W_ = sess.run(self.W)
            self.b_ = sess.run(self.b)
