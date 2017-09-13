import pandas as pd
import numpy as np
import os

def shuffle(X,Y):
    """ Shuffle two different numpy arrays according to the same indexes """
    assert len(X) == len(Y)
    randomize = np.arange(len(Y))
    np.random.shuffle(randomize)
    return X[randomize],Y[randomize]

class Dataset():
    """ Class to read and process the dataset """
    def __init__(self, path):
        #The path should be a folder with three different files:
        #labels.txt, titles.txt and label_titles.txt
        self.path = path
        self.type = path.replace("Datasets","")
        self.label_titles, self.labels, self.sentences = self.read()
        assert len(self.labels) == len(self.sentences)
        self.length = len(self.labels)
        self.divide()

    def read(self):
        """ Read the three files that compose the dataset """
        def read_txt(x):
            with open(x, "r") as text_to_read:
                return np.array([i[:-1] for i in text_to_read])
        label_titles = read_txt(self.path+"/label_titles.txt")
        labels = read_txt(self.path+"/labels.txt").astype(int)
        sentences = read_txt(self.path+"/titles.txt")
        sentences, labels = shuffle(sentences, labels)
        self.num_labels = np.max(labels)+1
        return label_titles, labels, sentences

    def divide(self):
        """ Divides the already read dataset into embedding, training and test set """
        titles, labels = self.sentences, self.labels
        #40% for embedding set, 40% for training set, 20% for test set
        #Note that to change the sizes you also need to make sure that no vocbulary files
        #(i.e Data_obtained/Dataset/Options) are left

        if os.path.exists("Data_obtained"+self.type):
            if os.path.isfile("Data_obtained"+self.type+"/embset.npy"):
                print("Separation into three sets already found")
                self.embset = np.load("Data_obtained"+self.type+"/embset.npy")
                self.trainset = np.load("Data_obtained"+self.type+"/trainset.npy")
                self.testset = np.load("Data_obtained"+self.type+"/testset.npy")
                return
        else:
            os.makedirs("Data_obtained"+self.type)
        size1 = self.length*4//10
        size2 = size1
        size3 = self.length - size1 - size2
        assert size1+size2+size3 == self.length
        print("Embedding set:", size1, "sentences")
        print("Training set:", size2, "sentences")
        print("Test set:", size3, "sentences")
        self.embset = np.array([titles[:size1],labels[:size1]])
        np.save("Data_obtained"+self.type+"/embset.npy", self.embset)
        self.trainset = np.array([titles[size1:size1+size2], labels[size1:size1+size2]])
        np.save("Data_obtained"+self.type+"/trainset.npy", self.trainset)
        self.testset = np.array([titles[size1+size2:], labels[size1+size2:]])
        np.save("Data_obtained"+self.type+"/testset.npy", self.testset)
