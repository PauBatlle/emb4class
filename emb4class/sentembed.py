""" Create embeddings for sentences from embeddings for words """
import numpy as np
import sklearn.preprocessing as pre
import re, string
from nltk import word_tokenize
from tqdm import tqdm


clean = lambda s: re.sub('['+string.punctuation.replace('+','').replace('/','')+']', '', s)
tokenizer = lambda x: word_tokenize(clean(x.lower()))

class sentence_embeddings():
    """ Class to store sentences_embeddings to feed them to the classifier """

    def __init__(self, data, embeddings):
        self.num2word = data.vocabulary
        self.word2num = {a:b for b,a in enumerate(data.vocabulary)}
        self.sentences_to_embed = data.separated_train
        clean = lambda x: [word for word in x if word in self.word2num.keys()]
        self.sentences_to_embed = np.array([clean(x) for x in self.sentences_to_embed])
        #We want to keep the sentences and labels from the training set that are not empty
        good_indexes = np.where(np.array([len(i) for i in self.sentences_to_embed]) > 0)[0]
        self.sentences_to_embed = self.sentences_to_embed[good_indexes]
        data.labels_train = data.trainset[1][good_indexes]
        assert min([len(i) for i in self.sentences_to_embed]) > 0
        self.embeddings = embeddings
        self.average()
        return

    def average(self):
        embe = []
        for sentence in tqdm(self.sentences_to_embed):
            sentence = [word for word in sentence if word in self.word2num.keys()]
            embe.append(sum([self.embeddings.bow[self.word2num[word]] for word in sentence]))
            #No need to divide by the average because we L2-Normalize
        self.bow_average = pre.normalize(np.array(embe))
