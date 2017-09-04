import numpy as np
import sklearn.preprocessing as pre
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(Y_test,Y_predicted,label_titles,file):
    """ Plot and save the confusion matrix """
    confusion_matrix(np.argmax(Y_test, axis = 1), np.argmax(Y_predicted, axis = 1)).astype(int)
    f, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", linewidths=.5, ax=ax, cmap = "jet", xticklabels= label_titles, yticklabels=label_titles)
    plt.save(file+".png")
    
def softmax(x):
    """ Compute softmax for a vector """
    e_x = np.exp(x-np.max(x))
    return e_x / np.sum(e_x)


class Evaluate():
    """Class to evaluate the embedding based on a classifier"""
    def __init__(data, sentembeddings, classifier):
        self.data = data
        self.sentembeddings = sentembeddings
        self.W = classifier.W_final
        self.b = classifier.b_final
        self.most_important_words()
        self.precision_recall()
        self.mean_average_precision()

    def onehot(self):
        y = self.data.labels_train.astype(int)
        encoded = []
        for i in y:
            vec = np.zeros(self.data.num_labels)
            vec[i] = 1
            encoded.append(vec)
        self.encoded_y_train = np.array(encoded)


    def most_important_words(self):
        self.X_test, self.y_test
        for i in range(len(W_.T)):
        print("*********")
        print(num2label[i+1])
        print("*********")
        fila = sorted(enumerate(softmax(W_.T[i])), key = lambda x: x[1])
        print("Les paraules menys importants de la categoria són: ")
        for i in fila[:5]:
            print ("\t", num2word[i[0]], i[1])
        print("Les paraules més importants de la categoria són: ")
        for i in fila[-5:]:
            print ("\t", num2word[i[0]], i[1])

    def precision_recall(self):
        """ Calculate the precision recall curve """
        v1 = np.array(np.argmax(Y_predicted, axis = 1) == np.argmax(Y_test, axis = 1), dtype = int)
        v2 = np.array([np.max(i) for i in Y_predicted])
        precision, recall, thresholds = precision_recall_curve(v1,v2)
        plt.plot(recall, precision
        data = np.max(Y_predicted, axis = 1)
        # evaluate the histogram
        values, base = np.histogram(data, bins=100)
        #evaluate the cumulative
        cumulative = np.cumsum(values)
        # plot the cumulative function
        plt.plot(base[:-1], 1-cumulative/len(data), c = 'blue')
        #plot the survival function

    def mean_average_precision(self):
        #Precision-Recall curves per class
        color_list = plt.cm.tab20(np.linspace(0, 1, 20))
        for i in range(20):
            if i == 0:
                plt.figure(figsize=(11,13))
            plt.title("Precision recall curves for each class")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            precision, recall, thresholds = precision_recall_curve(Y_test[:,i], Y_predicted[:,i])
            plt.plot(recall, precision, color = color_list[i])
            plt.legend(list(num2label.values()))
            sorted_AP = sorted([(num2label[i+1], average_precision_score(Y_test[:,i], Y_predicted[:,i])) for i in range(20)], key = lambda x: x[1], reverse = True)
            AP = [i[1] for i in sorted_AP]
            mAP = sum(AP)/len(AP)
            accuracy = np.zeros(20)
            for i in range(20):
            accuracy[i] = mat[i][i] / np.sum(mat[i])

    def confusion_matrix(self):
        plot_confusion_matrix(args)
