import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.preprocessing as pre
import tensorflow as tf

#We assume X_train, X_test already normalized
class Metric_learning():

    def __init__():
        pass

    def create_graph(self):
        self.qry = tf.placeholder(tf.float32, [1, dim])
        self.pos = tf.placeholder(tf.float32, [1, dim])
        self.neg = tf.placeholder(tf.float32, [1, dim])

        # The embedding parameters
        # Projection matrix
        self.W = tf.Variable(tf.random_normal([dim, hidden]))
        # We also add a bias term
        self.b = tf.Variable(tf.zeros([hidden]))

        # Function to embed the inputs
        self.eqry = tf.matmul(qry, W) + b
        self.epos = tf.matmul(pos, W) + b
        self.eneg = tf.matmul(neg, W) + b

        # Define triplet loss in the space of the embeddings
        self.cost = tf.maximum(0.0, 1 + tf.matmul(eqry, tf.transpose(tf.subtract(eneg, epos))))
        # Gradient Descent Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        return

    def train(self):
        # Launch the graph
        with tf.Session() as sess:
        # Initialize
        sess.run(tf.global_variables_initializer())
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            # Loop over all samples
            for i in np.arange(0, nsamples):
                # Each "input sample" consists of a "query", a positive and a negative example
                # This is constructed on the fly
                # Ideally, this should be precomputed
                # Query: take the next training example
                x_qry = x_train[i,None]
                # Positive: sample a training example from the same class as the query
                idx  = np.where(labels_train == labels_train[i])[0]
                r = np.random.randint(len(idx))
                ipos = idx[r]
                x_pos = x_train[ipos,None]
                # Negative: sample a training example with a different class label
                idx  = np.where(np.logical_not(labels_train == labels_train[i]))[0]
                r = np.random.randint(len(idx))
                ineg = idx[r]
                x_neg = x_train[ineg,None]
                # Perform gradient step
                _, c = sess.run([optimizer, cost], feed_dict={qry: x_qry,
                                                              pos: x_pos,
                                                              neg: x_neg})
                # Update loss of this epoch
                avg_cost += c
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", epoch+1, avg_cost/nsamples)
                What = sess.run(W)
                bhat = sess.run(b)
                eval_metric(What, bhat, epoch+1)
        self.W_ = sess.run(W)
        self.b_ = sess.run(b)
