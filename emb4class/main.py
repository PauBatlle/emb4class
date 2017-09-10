import numpy as np
import pandas as pd
from preprocess import Dataset
from Vocab import write_vocabulary
from embeddings import Embedding
from sentembed import sentence_embeddings
from classifier import Run_classifiers
from evaluate import Evaluate

class Vocabulary_options():
    """ Options taken into account to create the vocabulary """
    def __init__(self, stop_word_removal = False, lemmatizer = False, hard_lemmatizer = False, subsampling = False,
               subsampling_threshold = 1e-3, min_count = 2):
        self.stop_word_removal = stop_word_removal
        self.lemmatizer = lemmatizer
        self.hard_lemmatizer = hard_lemmatizer
        if (not self.lemmatizer):
            self.hard_lemmatizer = False
        self.subsampling = subsampling
        #It only gets applied if subsampling is True.
        #1e-5, the recommended value, is really small for such a small database! Should be around 1e-3/5e-3
        #It is not a good idea to use both subsampling and stop_word_removal due to loss of valuable words
        self.subsampling_threshold = subsampling_threshold

        #Should be at least 2, it avoids outlier in the dataset
        self.min_count = min_count
    def __str__(self):
        s = "MC="+str(self.min_count)+","
        s += "SWR=T," if self.stop_word_removal else "SWR=F,"
        s += "S=T" if self.subsampling else "S=F"
        s += str(round(self.subsampling_threshold,4))+"," if self.subsampling else ","
        s += "L=T" if self.lemmatizer else "L=F"
        s += "T" if self.hard_lemmatizer else ""
        return s

class General_options():
    """ General options regarding the classifier and some word embeddings """
    def __init__ (self, learning_rate = 0.01, training_epochs=2500, batch_size = 500,
        display_step = 500, window_size = 3, distribution_smoothing = 1, k = 5, embedding_size = 100):
        #Classifier options
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        #Embedding options
        self.window_size = window_size
        self.distribution_smoothing = distribution_smoothing
        self.k = k
        self.embedding_size = embedding_size

    def __str__(self):
        """String representation of the classifier options """
        return "LR="+str(self.learning_rate)+"-EPOCHS="+str(self.training_epochs)+"-BATCHSIZE="+str(self.batch_size)

class Metric_learning_options():
    """ Options to run the Metric Learning implementation """
    def __init__ (self, learning_rate = 0.01, training_epochs = 100, display_step = 10,
                embedding_size = 100):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.display_step = display_step
        self.embedding_size = embedding_size

    def __str__(self):
        return "LR="+str(self.learning_rate)+"-EPOCHS="+str(self.training_epochs)+"-EMSIZE="+str(self.embedding_size)

class Word2Vec_options():
    """ Options to run my implemented Word2Vec """
    def __init__(self, learning_rate = 1, total_batches = 7500, batch_size = 5000,
    display_step = 1000, embedding_size = 100, k = 5, iterations = 10):
        self.learning_rate = learning_rate
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.display_step = display_step
        self.embedding_size = embedding_size
        self.k = k
        #Number of iterations of Gensim's Word2Vec
        self.iterations = iterations

    def __str__(self):
        s = ""
        s+= "LR="+str(self.learning_rate)
        s+= "-BATCHES="+str(self.total_batches)
        s+= "-BATCHSIZE="+str(self.batch_size)
        s+= "-EMSIZE="+str(self.embedding_size)
        s+= "-K="+str(self.k)
        return s

class Experiment():
    def __init__(self, document_path, vocabulary_options, general_options,
                metric_learning_options, word2vec_options):
        self.vocabulary_options = vocabulary_options
        self.general_options = general_options
        self.metric_learning_options = metric_learning_options
        self.word2vec_options = word2vec_options
        """
        1.1-Read the dataset and split it into three groups:
        Embedding creation, classification test set, and classification evaluation set
        """
        print("Reading data")
        self.data = Dataset(document_path)
        """
        1.2- Create the vocabulary
        """
        print("-> Creating vocabulary if not found, reading it if found")
        write_vocabulary(self.data, self.vocabulary_options)
        """
        2.1-Generate all the word embeddings for the given dataset.
        """
        print("-> Generating word embeddings")
        self.embeddings = Embedding(self.data, self.general_options, self.metric_learning_options,
                                    self.word2vec_options)
        """
        2.2- From word embeddings to sentence embeddings
        """
        print("-> Generating sentence embeddings")
        self.sentembeddings = sentence_embeddings(self.data, self.embeddings, self.metric_learning_options, self.word2vec_options, self.general_options)
        """
        3 - Tensorflow classifier training
        """
        print("-> Training classifier")
        self.classifier = Run_classifiers(self.data,self.sentembeddings,self.general_options)
        """
        4 - Evaluation
        """
        print("-> Evaluating embeddings")
        #Since classifier already has self.data and self.embeddings, no need to pass them again
        Evaluate(self.classifier, self.general_options)

E = Experiment("Datasets/StackOverflow_20K", Vocabulary_options(), General_options(), Metric_learning_options(),Word2Vec_options())
