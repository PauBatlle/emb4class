import tensorflow as tf
import numpy as np
from preprocess import shuffle
from tqdm import tqdm
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
#Parameters
class Classifier():
    def __init__(self,data,sentembed,general_options):
        self.data = data
        self.sentembed = sentembed
        self.opt = general_options
        self.onehot()
        print("Creating graph")
        self.create_graph()
        print("Training classifier")
        self.train()

    def onehot(self):
        y = self.data.labels_train.astype(int)
        encoded = []
        for i in y:
            vec = np.zeros(self.data.num_labels)
            vec[i] = 1
            encoded.append(vec)
        self.encoded_y_train = np.array(encoded)

    def create_graph(self):
        """ Create the graph of a logistic regression classifier """
        self.nsamples, self.dim = self.sentembed.bow_average.shape
        self.num_classes = self.data.num_labels
        self.x = tf.placeholder(tf.float32, [None, self.dim], name = "x")
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name = "y")

        self.W = tf.Variable(tf.random_normal([self.dim, self.num_classes]))
        self.b = tf.Variable(tf.random_normal([self.num_classes]))

        self.aux = tf.matmul(self.x,self.W)+self.b
        self.yhat = tf.nn.softmax(self.aux, dim = 0)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.aux))
        self.optimizer = tf.train.GradientDescentOptimizer(self.opt.learning_rate).minimize(self.cost)
        self.correct_prediction = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype = tf.float32))

    def train(self):
        with tf.Session() as sess:
            X_train, y_train = self.sentembed.bow_average, self.encoded_y_train
            init = tf.global_variables_initializer()
            sess.run(init)
            costs = []
            acc = []
            for epoch in range(self.opt.training_epochs):
                X_train, y_train = shuffle(X_train, y_train)
                avg_cost = 0
                avg_acc = 0
                number_batches = int(self.nsamples/self.opt.batch_size)
                for i in range(number_batches):
                    batch_x = X_train[i*self.opt.batch_size:(i+1)*self.opt.batch_size]
                    batch_y = y_train[i*self.opt.batch_size:(i+1)*self.opt.batch_size]
                    c, _, acc = sess.run([self.cost, self.optimizer, self.accuracy], feed_dict = {self.x: batch_x, self.y: batch_y})
                    avg_cost += c/number_batches
                    avg_acc += acc/number_batches
                stop()
                if (epoch+1) % self.opt.display_step == 0:
                    print("Epoch {0}, cost {1:.5f}, average training accuracy {2:.5f}".format(str(epoch+1).zfill(4), avg_cost, avg_acc))
                costs += [avg_cost]
                acc += [avg_acc]
                plt.plot(costs)
                plt.savefig(self.data.directory+"costs.png")
                plt.close()
                plt.plot(acc)
                plt.savefig(self.data.directory+"acc.png")
                plt.close()
            self.W_final = sess.run(W)
            self.b_final = sess.run(b)
            np.save(self.data.directory+"W_final",self.W_final)
            np.save(self.data.directory+"B_final",self.b_final)
