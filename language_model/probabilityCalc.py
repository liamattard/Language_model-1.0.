import random
import numpy as np

from collections import Counter

def calculateProbabilityFromUnigram(unigram,sentence,word):
    return unigram[word]

def calculateProbabilityFromBigram(bigram,sentence,word):
    if (sentence[-1],word) in bigram:
        return bigram[sentence[-1],word]
    else:
        return 0

def calculateProbabilityFromTrigram(trigram,sentence,word):
    if (sentence[-2],sentence[-1],word) in trigram:
        return trigram[sentence[-2],sentence[-1],word]
    else: 
        return 0

def calculateProbabilityFromLaplaceBigram(trainCount,bigramCounts,sentence,lastWord):

    if (sentence[-1],lastWord) in bigramCounts:
        return (bigramCounts[sentence[-1],lastWord] +1)/(trainCount[sentence[-1]] + len(trainCount))
    else:
        if (sentence[-1] in trainCount) and (lastWord in trainCount):
            return 1/(trainCount[sentence[-1]] + len(trainCount))
        else:
            return 0

def calculateProbabilityFromLaplaceTrigram(trainCount, bigramCounts,trigramCounts,sentence,lastWord):

    if (sentence[-2],sentence[-1],lastWord) in trigramCounts:
        return  (trigramCounts[sentence[-2],sentence[-1],lastWord]+1)/(bigramCounts[sentence[-2],sentence[-1]] + len(trainCount))
    else:
        if ((sentence[-2],sentence[-1]) in bigramCounts) and (lastWord in trainCount):
            return 1/(bigramCounts[sentence[-2],sentence[-1]] + len(trainCount))
        else:
            return 0


def calculateProbabilityInterpolation(unigram,bigram,trigram,sentence,word):
    
    probabilityUnigram = 0.1* (unigram[word])
    probabilityBigram = 0.3 * (bigram[sentence[-1],word])
    probabilityTrigram = 0.6 * (trigram[sentence[-2],sentence[-1],word])
    return probabilityUnigram + probabilityBigram + probabilityTrigram

def calculateProbabilityLaplaceInterpolation(trainCount,unigramCount,bigramCounts,trigramCounts,sentence,lastWord):
    
    probabilityUnigram = 0.1 * (unigramCount[lastWord])
    probabilityBigram = 0.3 * (calculateProbabilityFromLaplaceBigram(trainCount,bigramCounts,sentence,lastWord))
    probabilityTrigram = 0.6 * (calculateProbabilityFromLaplaceTrigram(trainCount,bigramCounts,trigramCounts,sentence,lastWord))

    return probabilityUnigram + probabilityBigram + probabilityTrigram
