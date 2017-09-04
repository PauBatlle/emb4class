import numpy as np
import pandas as pd
#from Dataset2Vocab import Dataset, write_vocabulary
from preprocess import Dataset
from Vocab import write_vocabulary
from embeddings import Embedding
from sentembed import sentence_embeddings
from classifier import Classifier
from ipdb import set_trace as stop

class Vocabulary_options():
    """ Options to create the vocabulary """
    def __init__(self, stop_word_removal = False, lemmatizer = False, hard_lemmatizer = False, subsampling = False,
               subsampling_threshold = 1e-3, min_count = 2):
        self.stop_word_removal = stop_word_removal
        self.lemmatizer = lemmatizer
        self.hard_lemmatizer = hard_lemmatizer
        if (not self.lemmatizer):
            self.hard_lemmatizer = False
        self.subsampling = subsampling
        self.subsampling_threshold = subsampling_threshold
        #It only gets applied if subsampling is True.
        #1e-5, the recommended value, is really small for such a small database! Should be around 1e-3/5e-3
        #It is not a good idea to use both subsampling and stop_word_removal due to loss of valuable words
        self.min_count = min_count
        #Should be at least 2, it avoids outlier in the dataset

    def __str__(self):
        s = "MC="+str(self.min_count)+","
        s += "SWR=T," if self.stop_word_removal else "SWR=F,"
        s += "S=T" if self.subsampling else "S=F"
        s += str(round(self.subsampling_threshold,4))+"," if self.subsampling else ","
        s += "L=T" if self.lemmatizer else "L=F"
        s += "T" if self.hard_lemmatizer else ""
        return s

class General_options():
    def __init__ (self, learning_rate = 0.01, training_epochs=2500, batch_size = 100,
        display_step = 1):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step

class Metric_learning_options():
    def __init__ (self, learning_rate = 0.01, training_epochs = 100, display_step = 1):
        pass

class Experiment():
    def __init__(self, document_path, vocabulary_options, general_options):
        self.vocabulary_options = vocabulary_options
        self.general_options = general_options
        """
        1.1-Read the dataset and split it into three groups:
        Embedding creation, classification test set, and classification evaluation set
        """
        print("Reading data")
        self.data = Dataset(document_path)
        """
        1.2- Create the vocabulary
        """
        print("Creating vocabulary if not found, reading it if found")
        write_vocabulary(self.data, self.vocabulary_options)
        """
        2.1-Generate all the word embeddings for the given dataset.
        """
        print("Generating word embeddings")
        self.embeddings = Embedding(self.data)
        """
        2.2- From word embeddings to sentence embeddings
        """
        print("Generating sentence embeddings")
        self.sentembeddings = sentence_embeddings(self.data, self.embeddings)
        """
        3 - Tensorflow classifier training
        """
        print("Training classifier")
        self.classifier = Classifier(self.data,self.sentembeddings,self.general_options)
        """
        4 - Evaluation
        """
        self.Evaluate = Evaluate(self.data, self.sentembeddings, self.classifier)

E = Experiment("Datasets/StackOverflow_20K", Vocabulary_options(), General_options())
