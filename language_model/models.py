import numpy as np
from collections import Counter

def vanillaUnigram(train):
    '''
    Creates a vanilla unigram model of the training set with
    each word corresponding to its probability
    ex: {'word', 2}
    
    Parameters
    ----------
    train:numpy arr of training data from corpus

    Returns
    ----------
    unigram_model dict of words with probabilities
    '''
    unigram_model = Counter(train)

    for word in unigram_model:
        unigram_model[word] = unigram_model[word]/len(train)

    return unigram_model

def vanillaBigram(train):
    '''
    Creates a vanilla bigram model of the training set with
    each pair of words corresponding to its probability
    ex: { ('word','wordTwo'), 3}

    Parameters
    ----------
    train:numpy arr of training data from corpus

    Returns
    ----------
    bigram_model dict of pair of words with of probabilities

    '''
    bigram_model  = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])
    counter = Counter(train)

    for word in bigram_model:
        bigram_model[word] = bigram_model[word]/counter[word[0]]
    return bigram_model

def vanillaTrigram(train):
    '''
    Creates a vanilla tigram model of the training set with
    each trio of words corresponding to its probability
    ex: { ('word','wordTwo','wordThree'), 4}

    Parameters
    ----------
    train:numpy arr of training data from corpus

    Returns
    ----------
    bigram_model dict of trio of words with of probabilities

    '''
    bigram_model  = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])
    trigram_model = Counter([(word,train[i+1],train[i+2])for i,word in enumerate(train[:-2])])
    
    for word in trigram_model:
        trigram_model[word] = trigram_model[word]/bigram_model[(word[0],word[1])]
    return trigram_model

def laplaceUnigram(train):
    laplace_unigram = Counter(train)

    for word in laplace_unigram:
        laplace_unigram[word] = (laplace_unigram[word]+1)/len(train)

    return laplace_unigram

def unkUnigram(train):
    '''
    Creates a unkUnigram model of the training set with
    each trio of words corresponding to its probability
    ex: { ('word','wordTwo','wordThree'), 4}

    Parameters
    ----------
    train:numpy arr of training data from corpus

    Returns
    ----------
    bigram_model dict of trio of words with of probabilities

    '''

    counter = Counter(train)
    unigram_model ={}
    unigram_model["<UNK>"] = 0

    for word in counter:
        if counter[word] == 1:
            unigram_model["<UNK>"] += 1
        else:
            unigram_model[word] = counter[word]
        
    for word in unigram_model:
        unigram_model[word] = unigram_model[word]/len(train)
        
    return unigram_model

def unkBigram(train):
    unk_unigram = unkUnigram(train)
 
    for i,word in enumerate(train):
        if not (word in unk_unigram):
            train[i] = "<UNK>"
    

    return vanillaBigram(train)

def unkTrigram(train):
    unk_unigram = unkUnigram(train)

    for i,word in enumerate(train):
        if not (word in unk_unigram):
            train[i] = "<UNK>"
 
    return vanillaTrigram(train)

