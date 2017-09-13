""" Create embeddings for sentences from embeddings for words """
import numpy as np
import sklearn.preprocessing as pre
import re, string
from nltk import word_tokenize
from tqdm import tqdm
import os

clean = lambda s: re.sub('['+string.punctuation.replace('+','').replace('/','')+']', '', s)
tokenizer = lambda x: word_tokenize(clean(x.lower()))


class sentence_embeddings():
    """ Class to use the embeddings learned from the embedding set
        and using them to create embeddings in the test set
        for later feeding to the classifier, and do the same for the test set
        for later evaluation """

    def __init__(self, data, embeddings, metric_learning_options, word2vec_options, general_options):
        self.embeds = embeddings
        self.data = data
        self.mlearnopt = metric_learning_options
        self.w2vopt = word2vec_options
        self.genopt = general_options
        self.preprocess()
        self.average()


    def preprocess(self):
        """ Preprocess the training set and create a directory for
        saving the embeddings if necessary """
        if not os.path.exists(self.data.directory+"Sentence_embeddings"):
            os.makedirs(self.data.directory+"Sentence_embeddings")
            os.makedirs(self.data.directory+"Sentence_embeddings/Train")
            os.makedirs(self.data.directory+"Sentence_embeddings/Test")

        self.num2word = self.data.vocabulary
        self.word2num = {a:b for b,a in enumerate(self.num2word)}
        self.sentences_to_embed_train = self.data.separated_train
        self.sentences_to_embed_test = self.data.separated_test
        word_set = set(self.num2word)
        clean = lambda x: [word for word in x if word in word_set]
        self.sentences_to_embed_train = np.array([clean(x) for x in self.sentences_to_embed_train])
        self.sentences_to_embed_test = np.array([clean(x) for x in self.sentences_to_embed_test])
        #We want to keep the sentences and labels from the training set that are not empty
        good_indexes_train = np.where(np.array([len(i) for i in self.sentences_to_embed_train]) > 0)[0]
        good_indexes_test = np.where(np.array([len(i) for i in self.sentences_to_embed_test]) > 0)[0]

        self.sentences_to_embed_train = self.sentences_to_embed_train[good_indexes_train]
        self.sentences_to_embed_test = self.sentences_to_embed_test[good_indexes_test]
        
        self.labels_to_train = self.data.trainset[1][good_indexes_train]
        self.labels_to_test = self.data.testset[1][good_indexes_test]

        #Finally, check no empty sentence is left
        assert min([len(i) for i in self.sentences_to_embed_train]) > 0
        assert min([len(i) for i in self.sentences_to_embed_test]) > 0


    def average(self):
        """ Create the sentence embedding averaging the embedding of each word. Embeddings are also normalized """
        directory_train = self.data.directory+"Sentence_embeddings/Train/Averages/"
        directory_test = self.data.directory+"Sentence_embeddings/Test/Averages/"

        if not os.path.exists(directory_train):
            os.makedirs(directory_train)
        if not os.path.exists(directory_test):
            os.makedirs(directory_test)

        def create_embeddings(embedding_matrix, tokenized_sentences, word2num):
            """Create the embeddings"""
            sent_embeds = np.zeros((len(tokenized_sentences), len(embedding_matrix[0])))
            for number, sentence in enumerate(tokenized_sentences):
                s = sum([embedding_matrix[word2num[word]] for word in sentence])
                sent_embeds[number] = s/np.linalg.norm(s)
            return sent_embeds

        def check_create(name, embedding, tokenized_sentences = (self.sentences_to_embed_train,self.sentences_to_embed_test), word2num = self.word2num):
            """ Check whether the sentence embedding exists. If it does, read it,
            if it doesn't, call create_embdedings and create it """
            if os.path.isfile(directory_train+name+"-avg.npy"):
                return np.load(directory_train+name+"-avg.npy"), np.load(directory_test+name+"-avg.npy")
            else:
                 embs_train = create_embeddings(embedding, tokenized_sentences[0], word2num)
                 np.save(directory_train+name+"-avg", embs_train)
                 embs_test = create_embeddings(embedding, tokenized_sentences[1], word2num)
                 np.save(directory_test+name+"-avg", embs_test)
                 return embs_train, embs_test

        print("BoW")
        self.bowavg, self.bowavgtest = check_create("BoW", self.embeds.bow)

        print("PMI")

        self.pmi_uavg, self.pmi_uavgtest = check_create("PMIu", self.embeds.PMI_U)
        self.pmi_usavg, self.pmi_usavgtest = check_create("PMIus", self.embeds.PMI_Usigma)
        self.pmi_ussavg, self.pmi_ussavgtest = check_create("PMIuss", self.embeds.PMI_Ussigma)

        print("PPMI")

        self.ppmi_uavg, self.ppmi_uavgtest = check_create("PPMIu", self.embeds.PPMI_U)
        self.ppmi_usavg, self.ppmi_usavgtest = check_create("PPMIus", self.embeds.PPMI_Usigma)
        self.ppmi_ussavg, self.ppmi_ussavg = check_create("PPMIuss", self.embeds.PPMI_Ussigma)

        print("SPPMI")

        self.sppmi_uavg, self.sppmi_uavgtest = check_create("SPPMIu", self.embeds.SPPMI_U)
        self.sppmi_usavg, self.spmmi_usavg = check_create("SPPMIus", self.embeds.SPPMI_Usigma)
        self.sppmi_ussavg, self.sspmi_ussavg = check_create("SPPMIuss", self.embeds.SPPMI_Ussigma)

        print("Metric Learning")

        self.metricavg, self.metricavgtest = check_create("Metric_learning-"+str(self.mlearnopt), self.embeds.MetricLearningW)

        print("Word2Vecs")

        self.MyW2Vavg, self.MyW2Vavgtest = check_create("MyW2V-"+str(self.w2vopt), self.embeds.MyW2VVecs)

        embedding_size = self.w2vopt.embedding_size
        window = self.genopt.window_size
        iterations = self.w2vopt.iterations

        gensimoptstring = "EMSIZE="+str(embedding_size)+"-WINDOW="+str(window)+"-ITER="+str(iterations)

        self.GensimW2Vavg, self.GensimW2Vavgtest = check_create("GensimW2V-"+gensimoptstring, self.embeds.GensimVecs)
