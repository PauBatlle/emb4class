import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from IPython import embed
from tqdm import tqdm as t

class Embedding():
    ''' Class to store embeddings. Each embedding should be a matrix
        with shape len(vocabulary)*embedding_size '''
    def __init__(self, dataset, general_options):
        self.data = dataset
        self.embedding_directory = self.data.directory+"Embeddings/"
        self.pairs_directory = self.data.directory+"Pairs/"
        self.general_options = general_options
        if not os.path.exists(self.embedding_directory):
            os.makedirs(self.embedding_directory)
        if not os.path.exists(self.pairs_directory):
            os.makedirs(self.pairs_directory)

        self.context_word_pairs_creation()
        self.BoW()
        self.create_embedding_dictionary()
    def BOW(self):
        """Generate the Bag of Words Embeddings (or read them if already available)"""
        if os.path.isfile(self.embedding_directory+"BoW/embedding.npy"):
            self.bow = np.load(self.embedding_directory+"BoW/embedding.npy")
        else:
            if not os.path.exists(self.embedding_directory+"BoW"):
                os.makedirs(self.embedding_directory+"BoW")
            self.bow = np.eye(len(self.data.vocabulary))
            np.save(self.embedding_directory+"BoW/embedding", self.bow)

    def context_word_pairs_creation(self):
        """ Create context word pairs that will be used by some embeddings
         such as PMI and Word2Vec """
        self.pairs = {}
        window_sizes_to_try = self.general_options.window_sizes_to_try
        for window in window_sizes_to_try:
            if os.path.isfile(self.pairs_directory+str(window)+".npy"):
                #The pairs have already been calculated
                self.pairs[window] = np.load(self.pairs_directory+str(window)+".npy")
            else:
                word2num = {a:b for b,a in enumerate(self.data.vocabulary)}
                pairs = []
                if window == -1:
                    for sent in t(self.data.separated_emb):
                    aux = 0
                    for word in sent:
                        valid_indexs = [i for i in range(len(sent)) if i !=aux]
                        for num in valid_indexs:
                            pairs.append([word2num[s[num]], word2num[word]])
                        aux+=1
                else:
                    for sent in t(self.data.separated_emb):
                    aux = 0
                    for word in sent:
                        valid_indexs = [i for i in range(len(sent)) if abs(i-aux) <= window and i != aux]
                        for num in valid_indexs:
                            pairs.append([word2num[s[num]], word2num[word]])
                        aux+=1
    def Matrices(self):
        """ Generate the embeddings """

    def Metric_learning(self):
        pass

    def Word2Vec(self):
        pass
