import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Embedding():
    ''' Class to store embeddings. Each embedding should be a matrix
        with shape len(vocabulary)*embedding_size '''
    def __init__(self, dataset, general_options = None):
        self.data = dataset
        self.BOW()
        self.emblist = ["bow"]
        return
    def BOW(self):
        """Generate the simple BoW Embeddings """
        self.bow = np.eye(len(self.data.vocabulary))

    def Tfidf(self):
        pass

    def Word2Vec(self):
        pass
