import numpy as np
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from IPython import embed

def distance_matrix(embeddings, directory, name = ""):
    """ Plot the matrix distance between words, given an embedding
        (if the plot doesn't already exist) """
    if not os.path.isfile(directory+"/Similarities"+name+".npy"):
        M = pre.normalize(embeddings)
        S = np.dot(M,M.copy().T)
        plt.imshow(S)
        plt.colorbar()
        np.save(directory+"/Similarities"+name, S)
        plt.savefig(directory+"/sim_matrix"+name+".png")
        plt.close()


class Evaluate():
    def __init__(self, classifiers, general_options):
        self.classifiers = classifiers
        self.general_options = general_options
        self.data = classifiers.data
        self.labels = self.classifiers.sentembeddings.labels_to_test.astype(int)
        self.preprocess()
        self.onehot()
        self.read_directory = self.data.directory+"Classifiers/"+str(self.general_options)+"/"
        self.fully_evaluate("BoW", self.classifiers.sentembeddings.bowavgtest,
                            W = np.load(self.read_directory+"BoW/W.npy"), b = np.load(self.read_directory+"BoW/b.npy"))

    def preprocess(self):
        if not os.path.exists(self.data.directory+"Evaluation_results"):
            os.makedirs(self.data.directory+"Evaluation_results")
        self.write_directory = self.data.directory+"Evaluation_results/"

    def onehot(self):
        """ One-hot the labels """
        y = self.labels
        encoded = []
        for i in y:
            vec = np.zeros(self.data.num_labels)
            vec[i] = 1
            encoded.append(vec)
        self.labels = np.array(encoded)

    def fully_evaluate(self, name, embedded_test_set, W, b):
        X = embedded_test_set
        y = self.labels
        write_dir = self.write_directory+name+"/"
        num2word = self.data.vocabulary
        num2label = self.data.label_titles
        """Check if folder exists, and if not, create it"""
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        """Auxiliary functions"""

        def softmax(x):
            """Compute softmax for a vector """
            e_x = np.exp(x-np.max(x))
            return e_x / np.sum(e_x)

        """Accuracy"""
        aux = np.dot(X,W)+b
        y_hat = np.array([softmax(i) for i in aux])
        acc = np.mean(np.argmax(y_hat, axis = 1) == np.argmax(y, axis = 1))

        """Confusion matrix """
        M = confusion_matrix(np.argmax(y, axis = 1), np.argmax(y_hat, axis = 1), labels = range(len(num2label))).astype(int)
        f, ax = plt.subplots(figsize=(13, 13))
        sns.heatmap(M, annot=True, fmt="d", linewidths=.5, ax=ax, cmap = "jet", xticklabels=num2label, yticklabels=num2label)
        plt.savefig(write_dir+"confusion_matrix.png")
        plt.close()

        """Most important words"""
        with open(write_dir+"most_important_words.txt", "w") as file_to_write:
            p = file_to_write.write #shortcut
            for i in range(len(W.T)):
                p("*********"+'\n')
                p(num2label[i]+'\n')
                p("*********"+'\n')
                row = sorted(enumerate(softmax(W.T[i])), key = lambda x: x[1])
                p("The less important words of the category are: "+'\n')
                for i in row[:5]:
                    p('\t'+num2word[i[0]]+" "+str(i[1])+'\n')
                p("The most important words of the category are: "+'\n')
                for i in reversed(row[-5:]):
                    p('\t'+num2word[i[0]]+" "+str(i[1])+'\n')

        """Precision-recall"""
        v1 = np.array(np.argmax(y_hat, axis = 1) == np.argmax(y, axis = 1), dtype = int)
        v2 = np.array([np.max(i) for i in y_hat])
        precision, recall, thresholds = precision_recall_curve(v1,v2)
        plt.plot(recall, precision)
        plt.title("Precision-recall")
        plt.savefig(write_dir+"precision_recall.png")

        """Average-precision per class and mean average precision """

        color_list = plt.cm.tab20(np.linspace(0, 1, 20))
        for i in range(20):
            if i == 0:
                plt.figure(figsize=(11,13))
            plt.title("Precision recall curves for each class")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            precision, recall, thresholds = precision_recall_curve(y[:,i], y_hat[:,i])
            plt.plot(recall, precision, color = color_list[i])
            plt.legend(num2label)
            plt.savefig(write_dir+"ap_per_class.png")
            sorted_AP = sorted([(num2label[i], average_precision_score(y[:,i], y_hat[:,i])) for i in range(20)], key = lambda x: x[1], reverse = True)
            AP = [i[1] for i in sorted_AP]
            mAP = sum(AP)/len(AP)

        """ Accuracy for each class """
        accuracy = np.zeros(len(num2label))
        for i in range(len(num2label)):
            accuracy[i] = M[i][i] / np.sum(M[i])
        sorted_acc = sorted([(num2label[i], accuracy[i]) for i in range(20)], key = lambda x: x[1], reverse = True)

        """Write results to text file"""

        with open(write_dir+"metrics.txt", "w") as file_to_write:
            file_to_write.write("Acccuracy:"+str(acc)+"\n")
            file_to_write.write("mAP:"+str(mAP)+"\n")
            file_to_write.write("AP per class"+"\n")
            for label, score in sorted_AP:
                file_to_write.write("Class "+label+'\t'+str(score)+'\n')
            file_to_write.write("Accuracy per class"+"\n")
            for label, accuracy in sorted_acc:
                file_to_write.write("Class "+label+'\t'+str(accuracy)+'\n')
