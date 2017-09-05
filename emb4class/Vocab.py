import os
import pandas as pd
import numpy as np
import re
import string
from nltk import word_tokenize
import collections
from tqdm import tqdm


def write_vocabulary(dataset, options):
    """ Create the vocabulary according to the options if not already found. The vocabulary is created with the embedding part of the whole dataset ''"""
    dataset.directory = "Data_obtained"+dataset.type+"/"+str(options)+"/"
    if os.path.exists(dataset.directory):
        print("Vocabulary already found!")
        #Read from file
        dataset.frequences = np.load(dataset.directory+'counts.npy')
        dataset.vocabulary = dataset.frequences[:,0]
        dataset.separated_emb = np.load(dataset.directory+'separated_emb.npy')
        dataset.separated_train = np.load(dataset.directory+'separated_train.npy')
        dataset.separated_test = np.load(dataset.directory+'separated_test.npy')
        return
    else:
        print("Vocabulary not found, creating directory")
        os.makedirs(dataset.directory)
    #Tokenizer combining regexpr and nltk buit-in tokenizer
    clean = lambda s: re.sub('['+string.punctuation.replace('+','').replace('/','')+']', '', s)
    tokenizer = lambda x: word_tokenize(clean(x.lower()))
    #--> Min-count
    #Tokenize
    print("Tokenizing sentences")
    print("Embedding set")
    dataset.separated_emb = [tokenizer(i) for i in tqdm(dataset.embset[0])]
    print("Training set")
    dataset.separated_train = [tokenizer(i) for i in tqdm(dataset.trainset[0])]
    print("Test set")
    dataset.separated_test = [tokenizer(i) for i in tqdm(dataset.testset[0])]
    words = np.concatenate(dataset.separated_emb)
    #Count word occurences
    print("Counting word occurences")
    count = collections.Counter(words).most_common()
    #eliminate words appearing less than min_count times
    dataset.frequences = [w for w in count if w[1] >= options.min_count]
    dataset.vocabulary = [w[0] for w in dataset.frequences]
    #Convert to dictionary for easier access
    dataset.frequences = dict(dataset.frequences)
    print("Applying vocabulary options")
    """
    Apply min-count
    """
    words_to_keep = lambda x: [i for i in x if i in dataset.vocabulary]
    dataset.separated_emb = [words_to_keep(o) for o in dataset.separated_emb]
    """
    Apply lemmatizer
    """
    if options.lemmatizer:
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        pos = 'v' if options.hard_lemmatizer else 'n'
        lemmatize = lambda x: [wordnet_lemmatizer.lemmatize(i, pos = pos) for i in x]
        dataset.separated_emb = [lemmatize(u) for u in dataset.separated_emb]
    """
    Apply stop-word Removal
    """
    if options.stop_word_removal:
        from nltk.corpus import stopwords
        eliminated_words = set(stopwords.words('english'))
        eliminate = lambda x: [i for i in x if i not in eliminated_words]
        dataset.separated_emb = [eliminate(o) for o in dataset.separated_emb]
    """
    Apply subsampling
    """
    if options.subsampling:
        eliminated_words = []
        t = options.subsampling_threshold
        total = len(words)
        for pair in dataset.frequences.items():
            f = pair[1]/total
            p_removal =  1-(t/f)**0.5
            if np.random.rand() < p_removal:
                eliminated_words.append(pair[0])
        eliminate = lambda x: [i for i in x if i not in eliminated_words]
        dataset.separated_emb = [eliminate(o) for o in dataset.separated_emb]

    #Delete empty sentences
    dataset.separated_emb = [i for i in dataset.separated_emb if len(i) > 0]

    #If at least one of Lemmatizer, Stop-Word removal or subsampling was True, we need to recalculate counts
    if options.subsampling or options.stop_word_removal or options.lemmatizer:
        print("Recalculating word occurences")
        words = np.concatenate(dataset.separated_emb)
        count = collections.Counter(words).most_common()
        #eliminate words appearing less than min_count times
        dataset.frequences = [w for w in count if w[1] >= options.min_count]
        dataset.vocabulary = [w[0] for w in dataset.frequences]
        #Convert to dictionary for easier access
        dataset.frequences = dict(dataset.frequences)

    """ Write results to file, to avoid innecessary work in future executions """
    print("Writing to file")
    np.save(dataset.directory+'counts',np.array(list(dataset.frequences.items())))
    np.save(dataset.directory+'separated_emb', dataset.separated_emb)
    np.save(dataset.directory+'separated_train', dataset.separated_train)
    np.save(dataset.directory+'separated_test', dataset.separated_test)
    return
