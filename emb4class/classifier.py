import tensorflow as tf
import numpy as np
from preprocess import shuffle
from tqdm import tqdm as t
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
import os
from IPython import embed

class Run_classifiers():

    def __init__(self, data, sentembeddings, general_options):
        self.data = data
        self.sentembeddings = sentembeddings
        self.general_options = general_options
        self.labels = sentembeddings.labels_to_train.astype(int)
        self.num_labels = self.data.num_labels
        self.preprocess()
        self.onehot()
        self.one_run("BoW", self.sentembeddings.bowavg)
        self.classifiers_to_evaluate = ["BoW"]
    def preprocess(self):
        #Create directory for classification, if necessary
        if not os.path.exists(self.data.directory+"Classifiers"):
            os.makedirs(self.data.directory+"Classifiers")
        self.directory = self.data.directory+"Classifiers/"
        if not os.path.exists(self.directory+str(self.general_options)):
            os.makedirs(self.directory+str(self.general_options))
        self.directory = self.directory+str(self.general_options)+"/"

    def onehot(self):
        """ One-hot the labels """
        y = self.labels
        encoded = []
        for i in y:
            vec = np.zeros(self.num_labels)
            vec[i] = 1
            encoded.append(vec)
        self.Y = np.array(encoded)

    def one_run(self, name, embeddings):
        """ Check whether the results of the classifier exist and decide to read or train """
        if os.path.exists(self.directory+name) and os.path.isfile(self.directory+name+"/W.npy"):
            return
        os.makedirs(self.directory+name)
        model = Classifier(sentembed = embeddings, encoded_labels = self.Y, num_labels = self.num_labels,
                    directory = self.directory+name+"/", options = self.general_options)
        np.save(self.directory+name+"/W", model.W_final)
        np.save(self.directory+name+"/b", model.b_final)
        return



class Classifier():
    """ Run a logistic regression classifier for one of the embeddings """
    def __init__(self,sentembed,encoded_labels,num_labels,directory,options):
        self.X = sentembed
        self.Y = encoded_labels
        self.opt = options
        self.num_labels = num_labels
        self.directory = directory
        print("Creating graph")
        self.create_graph()
        print("Training classifier")
        self.train()

    def create_graph(self):
        """ Create the graph of a logistic regression classifier """
        self.nsamples, self.dim = self.X.shape
        self.num_classes = self.num_labels
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
        """ Train the model. Inputs: self.X, self.Y"""
        with tf.Session() as sess:
            X_train, y_train = self.X, self.Y
            init = tf.global_variables_initializer()
            sess.run(init)
            costs = []
            accuracy = []
            for epoch in t(range(self.opt.training_epochs)):
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
                if epoch % self.opt.display_step == 0:
                    print("Epoch {0}, cost {1:.5f}, average training accuracy {2:.5f}".format(str(epoch+1).zfill(4), avg_cost, avg_acc))
                costs += [avg_cost]
                accuracy += [avg_acc]
                plt.plot(costs)
                plt.title("Cost")
                plt.savefig(self.directory+"costs.png")
                plt.close()
                plt.plot(accuracy)
                plt.title("Accuracy (train set)")
                plt.savefig(self.directory+"acc.png")
                plt.close()
            self.W_final = sess.run(self.W)
            self.b_final = sess.run(self.b)
