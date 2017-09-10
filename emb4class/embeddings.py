import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.preprocessing as pre
from IPython import embed
from tqdm import tqdm as t
from metric_learning import Metric_Learning
from W2Vtf import Word2Vec
from ipdb import set_trace as stop
import gensim
from evaluate import distance_matrix


class Embedding():
    ''' Class to store embeddings. Each embedding should be a matrix
        with shape len(vocabulary)*embedding_size '''
    def __init__(self, dataset, general_options, metric_learning_options, word2vec_options):
        self.data = dataset
        self.embedding_directory = self.data.directory+"Embeddings/"
        self.pairs_directory = self.data.directory+"Pairs/"
        self.general_options = general_options
        self.metric_learning_options = metric_learning_options
        self.word2vec_options = word2vec_options

        if not os.path.exists(self.embedding_directory):
            os.makedirs(self.embedding_directory)
        if not os.path.exists(self.pairs_directory):
            os.makedirs(self.pairs_directory)

        print("Creating/Reading Word Context Pairs")
        self.Context_word_pairs_creation()
        print("Creating Counting Embeddings")
        self.BOW()
        self.Create_word_context_matrix()
        self.Matrices()
        self.Factorize_matrices()
        self.Metric_learning_embedding()
        self.Word2Vectf()
        self.GensimWord2Vec()
        print("Creating distance matrices if not found")
        self.distance_matrices()
        self.create_embedding_dict()

    def BOW(self):
        """Generate the Bag of Words Embeddings (or read them if already available)"""
        if os.path.isfile(self.embedding_directory+"BoW/embedding.npy"):
            self.bow = np.load(self.embedding_directory+"BoW/embedding.npy")
        else:
            if not os.path.exists(self.embedding_directory+"BoW"):
                os.makedirs(self.embedding_directory+"BoW")
            self.bow = np.eye(len(self.data.vocabulary))
            np.save(self.embedding_directory+"BoW/embedding", self.bow)
    def Context_word_pairs_creation(self):
        """ Create context word pairs that will be used by some embeddings
         such as PMI and Word2Vec """
        window = self.general_options.window_size
        if os.path.isfile(self.pairs_directory+str(window)+".npy"):
            print("Pairs already found!")
            #The pairs have already been calculated
            self.pairs = np.load(self.pairs_directory+str(window)+".npy")
        else:
            #Calculate them
            print("Context-word pairs not found, creating them")
            word2num = {a:b for b,a in enumerate(self.data.vocabulary)}
            pairs = []
            if window == -1:
                for sent in t(self.data.separated_emb):
                    aux = 0
                    for word in sent:
                        valid_indexs = [i for i in range(len(sent)) if i !=aux]
                        for num in valid_indexs:
                            pairs.append([word2num[sent[num]], word2num[word]])
                        aux+=1
                self.pairs[window] = pairs
                np.save(self.pairs_directory+str(window),pairs)
            else:
                for sent in t(self.data.separated_emb):
                    aux = 0
                    for word in sent:
                        valid_indexs = [i for i in range(len(sent)) if abs(i-aux) <= window and i != aux]
                        for num in valid_indexs:
                            pairs.append([word2num[sent[num]], word2num[word]])
                        aux+=1
                self.pairs = pairs
                np.save(self.pairs_directory+str(window),pairs)

    def Create_word_context_matrix(self):
        """ Create word-context matrix """
        window = self.general_options.window_size
        if os.path.isfile(self.pairs_directory+"WCM"+str(window)+".npy"):
            #The matrix has already been calculated
            print("Word-context Matrix already found!")
            self.WCMatrix = np.load(self.pairs_directory+"WCM"+str(window)+".npy")
        else:
            #Calculate it
            print("Word-context matrix not found, creating it")
            M = np.zeros((len(self.data.vocabulary), len(self.data.vocabulary)), dtype = np.float32)
            #Note that this matrix is word-context while the pairs are context-word
            for i in t(self.pairs):
                M[i[1],i[0]] += 1
            self.WCMatrix = M
            np.save(self.pairs_directory+"WCM"+str(window)+".npy", M)

    def Matrices(self):
        """Generate the PMI, PPMI, SPPMI matrices"""
        alpha = self.general_options.distribution_smoothing
        k = self.general_options.k
        #We first check whether some of the matrices already exist
        PMI_exists = os.path.isfile(self.embedding_directory+"Matrices/PMI"+str(alpha)+".npy")
        PPMI_exists = os.path.isfile(self.embedding_directory+"Matrices/PPMI"+str(alpha)+".npy")
        SPPMI_exists = os.path.isfile(self.embedding_directory+"Matrices/SPPMI"+str(alpha)+"-"+str(k)+".npy")

        if PMI_exists:
            print("PMI Found")
            self.PMI = np.load(self.embedding_directory+"Matrices/PMI"+str(alpha)+".npy")
        if PPMI_exists:
            print("PPMI Found")
            self.PPMI = np.load(self.embedding_directory+"Matrices/PPMI"+str(alpha)+".npy")
        if SPPMI_exists:
            print("SPPMI Found")
            self.SPPMI = np.load(self.embedding_directory+"Matrices/SPPMI"+str(alpha)+"-"+str(k)+".npy")

        if PMI_exists and PPMI_exists and SPPMI_exists:
            return
        if not os.path.exists(self.embedding_directory+"Matrices"):
            os.makedirs(self.embedding_directory+"Matrices")

        M = self.WCMatrix
        matrix_sum = np.sum(M)
        prob_word = lambda w: np.sum(M[w,:])/matrix_sum
        column_sum = lambda x: np.sum(M[:,x])
        def prob_context(c, alpha):
            if alpha == 1:
                return column_sum(c)/matrix_sum
            else:
                return (column_sum(c)**alpha)/np.sum([column_sum(p)**alpha for p in range(len(M))])
        prob_word_context = lambda w,c: M[w,c]/matrix_sum
        PMI_Matrix = np.zeros(M.shape)
        PPMI_Matrix = np.zeros(M.shape)
        SPPMI_Matrix = np.zeros(M.shape)
        PMI = lambda i,j: np.log(prob_word_context(i,j)/(prob_word(i)*prob_context(j,alpha)))
        PPMI = lambda i,j: max(0, np.log(prob_word_context(i,j)/(prob_word(i)*prob_context(j,alpha))+1e-8))
        print("Creating PMI/PPMI and SPPMI Matrices")
        log_k = np.log(k)
        for i in t(range(len(M))):
            for j in range(len(M)):
                if M[i,j] > 1e-10:
                    PMI_Matrix[i,j] = PMI(i,j)
                    PPMI_Matrix[i,j] = PPMI(i,j)
                    SPPMI_Matrix[i,j] = max(PMI_Matrix[i,j]-log_k,0)
        self.PMI = PMI_Matrix
        self.PPMI = PPMI_Matrix
        self.SPPMI = SPPMI_Matrix
        if not PMI_exists:
            np.save(self.embedding_directory+"Matrices/PMI"+str(alpha),PMI_Matrix)
        if not PPMI_exists:
            np.save(self.embedding_directory+"Matrices/PPMI"+str(alpha),PPMI_Matrix)
        if not SPPMI_exists:
            np.save(self.embedding_directory+"Matrices/SPPMI"+str(alpha)+"-"+str(k),SPPMI_Matrix)

    def Factorize_matrices(self):
        """Perform SVD factorization to the matrix to obtain the embeddings """
        embsize = self.general_options.embedding_size
        norm = pre.normalize
        PMI_exists = os.path.exists(self.embedding_directory+"Matrices/PMI")
        PPMI_exists = os.path.exists(self.embedding_directory+"Matrices/PPMI")
        SPPMI_exists = os.path.exists(self.embedding_directory+"Matrices/SPPMI")
        if PMI_exists:
            print("PMI Factorization found")
            self.PMI_U = np.load(self.embedding_directory+"Matrices/PMI/U.npy")
            self.PMI_Usigma = np.load(self.embedding_directory+"Matrices/PMI/Us.npy")
            self.PMI_Ussigma = np.load(self.embedding_directory+"Matrices/PMI/Uss.npy")
            """
            self.PMI_UV = np.load(self.embedding_directory+"Matrices/PMI/U+V.npy")
            self.PMI_USV = np.load(self.embedding_directory+"Matrices/PMI/U+sV.npy")
            self.PMI_USSV = np.load(self.embedding_directory+"Matrices/PMI/Us+sV.npy")
            """
        if PPMI_exists:
            print("PPMI Factorization found")
            self.PPMI_U = np.load(self.embedding_directory+"Matrices/PPMI/U.npy")
            self.PPMI_Usigma = np.load(self.embedding_directory+"Matrices/PPMI/Us.npy")
            self.PPMI_Ussigma = np.load(self.embedding_directory+"Matrices/PPMI/Uss.npy")
            """
            self.PPMI_UV = np.load(self.embedding_directory+"Matrices/PPMI/U+V.npy")
            self.PPMI_USV = np.load(self.embedding_directory+"Matrices/PPMI/U+sV.npy")
            self.PPMI_USSV = np.load(self.embedding_directory+"Matrices/PPMI/Us+sV.npy")
            """
        if SPPMI_exists:
            print("SPPMI Factorization found")
            self.SPPMI_U = np.load(self.embedding_directory+"Matrices/SPPMI/U.npy")
            self.SPPMI_Usigma = np.load(self.embedding_directory+"Matrices/SPPMI/Us.npy")
            self.SPPMI_Ussigma = np.load(self.embedding_directory+"Matrices/SPPMI/Uss.npy")
            """
            self.SPPMI_UV = np.load(self.embedding_directory+"Matrices/SPPMI/U+V.npy")
            self.SPPMI_USV = np.load(self.embedding_directory+"Matrices/SPPMI/U+sV.npy")
            self.SPPMI_USSV = np.load(self.embedding_directory+"Matrices/SPPMI/Us+sV.npy")
            """
        if PMI_exists and PPMI_exists and SPPMI_exists:
            return

        """For each of the three matrices (PMI, PPMI, SPPMI), factorize them and obtain
        three possible embeddings"""
        print("Factorizing matrices (1/3)")
        directory = self.embedding_directory+"Matrices/PMI/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        U, sigma, Vt = np.linalg.svd(self.PMI)
        U_trunc = U[:,:embsize]
        V_trunc = Vt[:embsize,:]
        sigma_trunc = np.diag(sigma[:embsize])
        sqrt_sigma_trunc = np.diag(np.sqrt(sigma[:embsize]))
        U = U_trunc
        Us = U_trunc@sigma_trunc
        Uss = U_trunc@sqrt_sigma_trunc
        np.save(directory+"U", U)
        np.save(directory+"Us", Us)
        np.save(directory+"Uss",Uss)
        self.PMI_U = U
        self.PMI_Usigma = Us
        self.PMI_Ussigma = Uss
        """
        np.save(directory+"U+V", norm(U_trunc+V_trunc.T))
        np.save(directory+"U+sV", norm(U_trunc@sigma_trunc + V_trunc.T))
        np.save(directory+"Us+sV", norm((U_trunc@sqrt_sigma_trunc)+((sqrt_sigma_trunc@V_trunc).T)))
        """
        print("Factorizing matrices (2/3)")
        directory = self.embedding_directory+"Matrices/PPMI/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        U, sigma, Vt = np.linalg.svd(self.PPMI)
        U_trunc = U[:,:embsize]
        V_trunc = Vt[:embsize,:]
        sigma_trunc = np.diag(sigma[:embsize])
        sqrt_sigma_trunc = np.diag(np.sqrt(sigma[:embsize]))
        U = U_trunc
        Us = U_trunc@sigma_trunc
        Uss = U_trunc@sqrt_sigma_trunc
        np.save(directory+"U", U)
        np.save(directory+"Us", Us)
        np.save(directory+"Uss",Uss)
        self.PPMI_U = U
        self.PPMI_Usigma = Us
        self.PPMI_Ussigma = Uss
        """
        np.save(directory+"U+V", norm(U_trunc+V_trunc.T))
        np.save(directory+"U+sV", norm(U_trunc@sigma_trunc + V_trunc.T))
        np.save(directory+"Us+sV", norm((U_trunc@sqrt_sigma_trunc)+((sqrt_sigma_trunc@V_trunc).T)))
        """
        print("Factorizing matrices (3/3)")
        directory = self.embedding_directory+"Matrices/SPPMI/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        U, sigma, Vt = np.linalg.svd(self.SPPMI)
        U_trunc = U[:,:embsize]
        V_trunc = Vt[:embsize,:]
        sigma_trunc = np.diag(sigma[:embsize])
        sqrt_sigma_trunc = np.diag(np.sqrt(sigma[:embsize]))
        U = U_trunc
        Us = U_trunc@sigma_trunc
        Uss = U_trunc@sqrt_sigma_trunc
        np.save(directory+"U", U)
        np.save(directory+"Us", Us)
        np.save(directory+"Uss",Uss)
        self.SPPMI_U = U
        self.SPPMI_Usigma = Us
        self.SPPMI_Ussigma = Uss
        """
        np.save(directory+"U+V", norm(U_trunc+V_trunc.T))
        np.save(directory+"U+sV", norm(U_trunc@sigma_trunc + V_trunc.T))
        np.save(directory+"Us+sV", norm((U_trunc@sqrt_sigma_trunc)+((sqrt_sigma_trunc@V_trunc).T)))
        """

    def Metric_learning_embedding(self):
        """ Read/create a Metric-Learned based embedding starting from simple BoW """
        directory = self.embedding_directory+"Metric_Learning/"+str(self.metric_learning_options)+"/"
        if os.path.exists(directory):
            print("Metric Learning with the selected options already calculated, reading results")
            self.MetricLearningW = np.load(directory+"W.npy")
            self.MetricLearningb = np.load(directory+"b.npy")
            return
        if not os.path.exists(self.embedding_directory+"Metric_Learning"):
            os.makedirs(self.embedding_directory+"Metric_Learning")
        os.makedirs(directory)
        print("Training Metric Learning")
        #Preprocessing for metric learning
        labels = self.data.embset[1].astype(int)
        separated_sentences = self.data.separated_emb
        non_empty_indexes = np.where(np.array([len(i) for i in separated_sentences]) != 0)[0]
        word2num = {a:b for b,a in enumerate(self.data.vocabulary)}
        separated_sentences, labels = separated_sentences[non_empty_indexes], labels[non_empty_indexes]
        #Changing this function from another can make Metric Learnig start from another embedding
        def bag_of_words(sentence):
            vec = sum([self.bow[word2num[i]] for i in sentence])
            normalized_vec = vec/np.linalg.norm(vec)
            return normalized_vec
        embedded_sentences = np.array([bag_of_words(i) for i in separated_sentences])
        M = Metric_Learning(options = self.metric_learning_options, embedded_sentences = embedded_sentences, labels = labels, directory = directory)
        #Save in-class and to file
        self.MetricLearningW = M.W_
        self.MetricLearningb = M.b_
        np.save(directory+"W",self.MetricLearningW)
        np.save(directory+"b",self.MetricLearningb)

    def Word2Vectf(self):
        """ Use my own implementation of Word2Vec (slow and inefficient) """
        #Ho he de retornar com self.MyW2VW
        directory = self.embedding_directory+"MyW2V/"+str(self.word2vec_options)+"/"
        if os.path.exists(directory):
            print("Own W2V with the selected options already calculated, reading results")
            self.MyW2VVecs = np.load(directory+"Vecs.npy")
            return
        if not os.path.exists(self.embedding_directory+"MyW2V"):
            os.makedirs(self.embedding_directory+"MyW2V")
        #Create directory with the options
        os.makedirs(directory)
        winsize = self.general_options.window_size
        pairs= np.load(self.pairs_directory+str(winsize)+".npy")
        print("Training Word2Vec")
        Model = Word2Vec(pairs = pairs, word2vec_options = self.word2vec_options, directory = directory)
        self.MyW2VVecs = Model.trained_embeddings
        np.save(directory+"Vecs.npy", self.MyW2VVecs)

    def GensimWord2Vec(self):
        embedding_size = self.word2vec_options.embedding_size
        sentences = self.data.separated_emb
        window = self.general_options.window_size
        iterations = self.word2vec_options.iterations
        self.gensimoptstring = options_string = "EMSIZE="+str(embedding_size)+"-WINDOW="+str(window)+"-ITER="+str(iterations)
        directory = self.embedding_directory+"GensimW2V/"+options_string+"/"
        if os.path.exists(directory):
            print("Gensim's W2V with the selected options already calculated, reading results")
            self.GensimVecs = np.load(directory+"Vecs.npy")
            self.gensimw2v = gensim.models.Word2Vec.load(directory+"model")
            return
        if not os.path.exists(self.embedding_directory+"GensimW2V"):
            os.makedirs(self.embedding_directory+"GensimW2V")
        os.makedirs(directory)
        print("Training Gensim's W2V")
        #Preprocessing for metric learning
        self.gensimw2v = gensim.models.Word2Vec(sentences = sentences, size = embedding_size , window = window, min_count=1, sample = 0, iter = iterations)
        self.gensimw2v.save(directory+"model")
        assert set(self.gensimw2v.wv.vocab.keys()) == set(self.data.vocabulary)
        M = np.array([self.gensimw2v.wv[word] for word in self.data.vocabulary])
        self.GensimVecs = M
        np.save(directory+"Vecs",M)

    def distance_matrices(self):

        distance_matrix(self.bow, self.embedding_directory+"BoW/")

        distance_matrix(self.PMI_U,self.embedding_directory+"Matrices/PMI/", "U")
        distance_matrix(self.PMI_Usigma,self.embedding_directory+"Matrices/PMI/", "Us")
        distance_matrix(self.PMI_Ussigma,self.embedding_directory+"Matrices/PMI/", "Uss")
        distance_matrix(self.PPMI_U,self.embedding_directory+"Matrices/PPMI/", "U")
        distance_matrix(self.PPMI_Usigma,self.embedding_directory+"Matrices/PPMI/", "Us")
        distance_matrix(self.PPMI_Ussigma,self.embedding_directory+"Matrices/PPMI/", "Uss")
        distance_matrix(self.SPPMI_U,self.embedding_directory+"Matrices/SPPMI/", "U")
        distance_matrix(self.SPPMI_Usigma,self.embedding_directory+"Matrices/SPPMI/", "Us")
        distance_matrix(self.SPPMI_Ussigma,self.embedding_directory+"Matrices/SPPMI/", "Uss")
        distance_matrix(self.MetricLearningW,self.embedding_directory+"Metric_Learning/"+str(self.metric_learning_options)+"/")
        distance_matrix(self.MyW2VVecs, self.embedding_directory+"MyW2V/"+str(self.word2vec_options)+"/")
        distance_matrix(self.GensimVecs, self.embedding_directory+"GensimW2V/"+self.gensimoptstring+"/")

    """Important: Should someone add an embedding to this pipeline, it should also
        be included in the dictionary created by the following function in order to be
        evaluated later """
    def create_embedding_dict(self):
        """ Create a dictionary for easier access to all of the embeddings """
        self.embed_dict = {'BoW':self.bow, 'PMI_U': self.PMI_U, 'PMI_Us': self.PMI_Usigma,
        'PMI_Uss': self.PMI_Ussigma, 'PPMI_U': self.PPMI_U, 'PPMI_Us': self.PPMI_Usigma,
        'PPMI_Uss': self.PPMI_Ussigma, 'SPPMI_U': self.SPPMI_U, 'SPPMI_Us': self.SPPMI_Usigma,
        'SPPMI_Uss': self.SPPMI_Ussigma, 'GensimW2V':self.GensimVecs, 'MyW2V':self.MyW2VVecs}
