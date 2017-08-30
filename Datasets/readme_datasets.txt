We provide four different short text classification datasets to try the package.
In all the datasets, we separate between the sentences and their category/label.
The labels are encoded as numbers from 0 to number_of_classes-1, where 0 is the first
label in the label_titles file and number_of_classes-1 is the last label in the same file.

The included datasets are only the parts of the original datasets which are relevant
for the short text classification task.

Datasets:

1) Short StackOverflow Dataset

20000 questions from Stack Overflow labelled as part of one of 20 different balanced categories.

The original dataset can be found here: https://github.com/jacoxu/StackOverflow
2015NAACL VSM-NLP workshop-"Short Text Clustering via Convolutional Neural Networks"

2) Barcelona Open Data building names

Extract of the dataset of Barcelona buildings and shops. For the classification purpose,
only the names of the commerces of some specific categories were used. As this is a real
dataset, there are some categories which are more common than others, although the less
common categories on the original dataset were omitted in the creation of this one.

There are about 35000 buildings and shops labelled in 21 different categories.

The full, original dataset can be found here:
http://opendata-ajuntament.barcelona.cat/data/es/dataset/cens-activitats-comercials

3) Answer types classification

Extract of the datasets used for the paper Xin Li, Dan Roth, Learning Question Classifiers. COLING'02 (2002).
The labels of the questions refer to the type of answer that someone would give to a particular question.
There are around 94000 questions and 5 categories. While not exactly evenely distributed, the categories 
are fairly balanced.

The original datasets can be found here:
http://cogcomp.org/Data/QA/QC/


4) Yahoo! Answers Topic Classification Dataset

Extract of The Yahoo! Answers topic classification dataset created by Xiang Zhang
(xiang.zhang@nyu.edu), used in the following paper
Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification.
Advances in Neural Information Processing Systems 28 (NIPS 2015).

There are 10 large main categories, and 1.460.000 question titles labeled in this balanced dataset.
